# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import time
from typing import Dict, List, Optional
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import heapq

from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.ngram_repeat_block import NGramRepeatBlock


class SequenceGenerator(nn.Module):
    def __init__(
            self,
            models,
            tgt_dict,
            beam_size=1,
            max_len_a=0,
            max_len_b=200,
            max_len=0,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=None,
            eos=None,
            symbols_to_strip_from_output=None,
            lm_model=None,
            lm_weight=1.0,
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence)
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.model.set_decoder_beam_size(self.beam_size)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.max_len = max_len or self.model.max_decoder_positions()

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len

        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
                hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    # TODO(myleott): unused, deprecate after pytorch-translate migration
    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(
            self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs
    ) -> List[List[Dict[str, Tensor]]]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def _generate_with_constraints_early_stop(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            constraints: List[List[List]] = None,
            # constraints[i][j][k]: the k-th token in the j-th constraints of sent[i], support bos in first constraint and eos in last constraint
            bos_token: Optional[int] = None,
            patience=10,
            return_time_ignore = False,
            src_dict=None,
            tgt_dict=None
    ):
        bos_token = self.eos if bos_token is None else bos_token
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception(
                "expected src_tokens or source in net input. input keys: "
                + str(net_input.keys())
            )
        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
            encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        # scores_full = (
        #     torch.zeros(bsz * beam_size, max_len + 1).to(scores)
        # )
        scores_full_indices = torch.arange(max_len + 1).repeat(bsz * beam_size, 1).to(new_order)
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(src_tokens)
                .long()
                .fill_(self.pad)
        )  # +2 for eos and pad
        tokens_indices = torch.arange(-1, max_len+1).repeat(bsz * beam_size, 1).to(scores_full_indices)
        # tokens[:, 0] = bos_token
        attn: Optional[Tensor] = None
        # constraints: bsz*C*L
        if constraints is None:
            constraints = [[] for _ in range(bsz)]
        prefix_constraints = np.repeat(
            np.array(
                [c_list[0] if len(c_list) > 0 and c_list[0][0] == bos_token else [bos_token] for c_list in constraints]),
            beam_size, axis=0)
        prefix_constraints = [c_list[0] if len(c_list) > 0 and c_list[0][0] == bos_token else c_list for c_list in constraints]
        prefix_constraints_array = np.empty((len(prefix_constraints),), dtype=object)
        prefix_constraints_array[...] = prefix_constraints
        # print(suffix_constraints_array, flush=True)
        prefix_constraints = np.repeat(prefix_constraints_array, beam_size, axis=0)

        suffix_constraints = [c_list[1:] if len(c_list) > 0 and c_list[0][0] == bos_token else c_list for c_list in constraints]
        suffix_constraints = [c_list if len(c_list) > 0 and c_list[-1][-1] == self.eos else (c_list + [[self.eos]]) for c_list in suffix_constraints]
        suffix_constraints_array = np.empty((len(suffix_constraints),), dtype=object)
        suffix_constraints_array[...] = suffix_constraints
        # print(suffix_constraints_array, flush=True)
        suffix_constraints = np.repeat(suffix_constraints_array, beam_size, axis=0)
        start_steps = torch.Tensor([len(x)-1 for x in prefix_constraints]).to(tokens)
        steps = start_steps+1
        for i, prefix in enumerate(prefix_constraints):
            prefix_token_num = len(prefix)
            tokens[i, : prefix_token_num] = torch.Tensor(prefix).to(tokens)
        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step
        finalized_heap = [[(-np.inf, -1, -1, {"tokens": "</s>"}) for j in range(beam_size)] for i in range(bsz)]
        for hp in finalized_heap:
            heapq.heapify(hp)
        heap_no_change_steps = torch.zeros(bsz).type_as(src_tokens).to(src_tokens.device)
        active_bsz_idx = torch.arange(bsz).type_as(src_tokens).to(src_tokens.device)
        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
                .unsqueeze(1)
                .type_as(tokens)
                .to(src_tokens.device)
        )
        cand_bbsz_offsets = (
            (torch.arange(0, bsz) * cand_size)
                .unsqueeze(1)
                .type_as(tokens)
                .to(src_tokens.device)
        )
        active_hypos_cand_idx = torch.arange(0, beam_size).repeat(bsz, 1).type_as(tokens).to(src_tokens.device).add(cand_bbsz_offsets).view(-1)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)
        lprobs_offsets = (
            (torch.arange(0, bsz) * beam_size * self.vocab_size)
                .unsqueeze(1)
                .type_as(tokens)
                .to(src_tokens.device)
        )

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None
        start_step = True
        times_to_ignore = time.time()
        while True:
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            time_start = time.time()

            current_max_step = steps.max()
            previous_step_mask = torch.arange(current_max_step).repeat(bsz * beam_size, 1).to(steps).lt((steps-1).unsqueeze(-1))
            current_steps_mask = torch.arange(current_max_step).repeat(bsz * beam_size, 1).to(steps).eq((steps-1).unsqueeze(-1))
            with torch.autograd.profiler.record_function(
                    "EnsembleModel: forward_decoder"
            ):
                # print('input tokens: ', tokens[:, :current_max_step], flush=True)
                lprobs, avg_attn_scores = self.model.forward_decoder(
                    tokens[:, : current_max_step],
                    encoder_outs,
                    None,
                    self.temperature,
                    return_all_prob=True
                )
            current_step_lprobs = lprobs[current_steps_mask]
            previous_tokens = tokens[:, 1:current_max_step]
            previous_steps_lprobs = torch.mul(lprobs, previous_step_mask.unsqueeze(-1))[:, : current_max_step - 1, :].gather(-1, previous_tokens.unsqueeze(-1)).squeeze(-1)  # set probs of future steps to 0

            scores[:, :current_max_step-1] = previous_steps_lprobs.cumsum(dim=1)
            # print("current_step_lprobs: ", current_step_lprobs.size(), current_step_lprobs, flush=True)
            # print("scores[:, :current_max_step-1]", scores[:, :current_max_step-1].size(), scores[:, :current_max_step-1], flush=True)
            # print("scores[:, current_max_step-1: current_max_step]", scores[:, current_max_step-1: current_max_step].size(), scores[:, current_max_step-1: current_max_step], flush=True)
            lprobs = current_step_lprobs + scores[:, current_max_step - 1:current_max_step]  # sequence probs for tokens[:,1:steps]+candidates at steps

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty
            lprobs[:,
            self.eos] = -math.inf  # never select eos in constrained decoding. Instead, stop decoding by maximize overall sequence prob

            time_end = time.time()
            times_to_ignore += time_end - time_start
            # # handle max length constraint
            # if step >= max_len:
            #     lprobs[:, : self.eos] = -math.inf
            #     lprobs[:, self.eos + 1:] = -math.inf
            # Shape: (batch, cand_size)

            # search.step
            lprobs = lprobs.view(bsz,beam_size,self.vocab_size)
            if start_step:
                # at the first step all hypotheses are equally likely, so use
                # only the first beam
                lprobs = lprobs[:, ::beam_size, :].contiguous()
                start_step = False
            top_prediction = torch.topk(
                lprobs.view(bsz, -1),
                k=min(
                    # Take the best 2 x beam_size predictions. We'll choose the first
                    # beam_size of these which don't predict eos to continue with.
                    beam_size * 2,
                    lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
                ),
            )
            # cand_scores = lprobs.view(-1)[cand_indices.add(lprobs_offsets).view(-1)].view(bsz, -1)

            cand_scores = top_prediction[0].view(bsz, -1)
            cand_indices = top_prediction[1].view(bsz, -1)
            cand_beams = torch.div(cand_indices, self.vocab_size, rounding_mode="trunc")
            cand_indices = cand_indices.fmod(self.vocab_size)

            # At this point, beams_buf and indices_buf are single-dim and contain relative indices
            # cand_scores, cand_indices, cand_beams = self.search.step(
            #     current_max_step,
            #     lprobs.view(bsz, -1, self.vocab_size),
            #     scores.view(bsz, beam_size, -1)[:, :, :current_max_step],
            #     tokens[:, : current_max_step + 1],
            #     original_batch_idxs,
            # )

            # cand_bbsz_idx contains beam indices for the top candidate
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_steps = steps[cand_bbsz_idx.view(-1)]
            cand_start_steps = start_steps[cand_bbsz_idx.view(-1)]
            # print('suffix_constraints: ', suffix_constraints, flush=True)
            # print('cand_bbsz_idx: ', cand_bbsz_idx, flush=True)
            cand_suffix_constraints = [list(itertools.chain(*x)) for x in suffix_constraints[cand_bbsz_idx.view(-1).cpu()]]
            cand_suffix_lens = torch.Tensor([len(x) for x in cand_suffix_constraints]).to(steps)
            cand_suffix_constraints_padded = torch.nn.utils.rnn.pad_sequence([torch.Tensor(x) for x in cand_suffix_constraints], batch_first=True, padding_value=self.pad).to(tokens)
            tokens_with_suffix_constraints = torch.cat([tokens[cand_bbsz_idx.view(-1), :current_max_step], cand_indices.view(-1).unsqueeze(-1), cand_suffix_constraints_padded], dim=-1)
            tokens_with_suffix_constraints_real_lens = cand_steps + 1 + cand_suffix_lens # +1 for bos token
            # print('cand_steps: ', cand_steps)
            # print('cand_suffix_lens: ', cand_suffix_lens)
            # print('tokens_with_suffix_constraints_real_lens 1: ', tokens_with_suffix_constraints_real_lens, flush=True)
            # print('tokens_with_suffix_constraints 1: ', tokens_with_suffix_constraints, flush=True)
            tokens_with_suffix_constraints = torch.nn.utils.rnn.pad_sequence(torch.split(tokens_with_suffix_constraints[tokens_with_suffix_constraints.ne(self.pad)], tokens_with_suffix_constraints_real_lens.tolist()),batch_first=True, padding_value=self.pad).to(tokens)
            # print('tokens_with_suffix_constraints 2: ', tokens_with_suffix_constraints, flush=True)
            current_max_len = (tokens_with_suffix_constraints_real_lens - 1).max()
            # forward for tokens with all suffix
            with torch.autograd.profiler.record_function(
                    "EnsembleModel: forward_decoder"
            ):
                lprobs_with_suffix, avg_attn_scores_with_suffix = self.model.forward_decoder(
                    tokens_with_suffix_constraints,
                    encoder_outs,
                    None,
                    self.temperature,
                    return_all_prob=True
                )
                # print('avg_attn_scores_with_suffix.shape: ', avg_attn_scores_with_suffix.size(), flush=True)
                avg_attn_scores_with_suffix = avg_attn_scores_with_suffix.view(bsz * cand_size, -1)
                length_mask = torch.arange(current_max_len + 1).repeat(tokens_with_suffix_constraints_real_lens.size(0), 1).to(steps).lt(
                    (tokens_with_suffix_constraints_real_lens-1).unsqueeze(-1))
                output_tokens = tokens_with_suffix_constraints[:, 1:] # has eos but no bos
                lprobs_output_tokens = torch.mul(lprobs_with_suffix, length_mask.unsqueeze(-1))[:, :-1, :].gather(-1, output_tokens.unsqueeze(-1)).squeeze(-1)  # set probs of future steps to 0

            cand_scores_avg = lprobs_output_tokens.sum(dim=1).div((tokens_with_suffix_constraints_real_lens-1) ** self.len_penalty)

            top_cands_scores, top_cands_indices = torch.topk(cand_scores_avg.reshape(bsz, cand_size), k=beam_size)
            attn_clone = avg_attn_scores_with_suffix.view(bsz, cand_size, -1)
            for i in range(bsz):
                pushed_flag = False
                original_bsz_idx = active_bsz_idx[i]
                for j in range(beam_size):
                    cand_score, cand_indice = top_cands_scores[i][j], top_cands_indices[i][j]
                    cand_bbsz_indice = i*beam_size + cand_indice
                    cand_tokens = output_tokens.view(bsz, cand_size, -1)[i, cand_indice,:]
                    cand_tokens_len = (tokens_with_suffix_constraints_real_lens - 1).view(bsz, cand_size)[i,cand_indice]
                    cand_tokens = cand_tokens[: cand_tokens_len]
                    cand_score_list = scores.index_select(0, cand_bbsz_idx.view(-1))[cand_bbsz_indice, :cand_tokens_len]
                    cand_step = cand_steps[cand_bbsz_indice]

                    cand_info={
                        "tokens": cand_tokens,
                        "score": cand_score,
                        "attention": attn_clone[i,cand_indice,:],
                        "alignment": torch.empty(0),
                        "positional_scores": cand_score_list[1:]-cand_score_list[:-1],
                        "scores": cand_score_list
                    }
                    cand = (cand_score, cand_step, j, cand_info)
                    pop_cand = heapq.heappushpop(finalized_heap[original_bsz_idx], cand)
                    if pop_cand[:3] != cand[:3]:
                        pushed_flag = True
                        del pop_cand
                    else:
                        del pop_cand
                        break
                if pushed_flag:
                    heap_no_change_steps[original_bsz_idx] = 0
                else:
                    heap_no_change_steps[original_bsz_idx] += 1
            # construct batch_idxs which holds indices of batches to keep for the next pass
            batch_mask = heap_no_change_steps[active_bsz_idx].lt(patience)
            if not all(batch_mask):
                new_bsz = sum(batch_mask)
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                cand_scores_avg = cand_scores_avg.view(bsz, cand_size)[batch_idxs].view(-1)
                # cand_scores_next_step_mask = cand_scores_next_step_mask.view(bsz, cand_size, -1)[batch_idxs].view(
                #     new_bsz * cand_size, -1)
                cand_steps = cand_steps.view(bsz, cand_size)[batch_idxs].view(-1)
                cand_start_steps = cand_start_steps.view(bsz, cand_size)[batch_idxs].view(-1)

                output_tokens = output_tokens.view(bsz, cand_size, -1)[batch_idxs].view(new_bsz * cand_size, output_tokens.shape[-1])
                batch_idxs_cpu = batch_idxs.cpu()
                prefix_constraints = prefix_constraints.reshape(bsz, beam_size)[batch_idxs_cpu].reshape(-1)
                # print('suffix_constraints 1:', suffix_constraints, flush=True)
                # print('batch_idxs_cpu: ', batch_idxs_cpu)
                suffix_constraints = suffix_constraints.reshape(bsz, beam_size)[batch_idxs_cpu].reshape(-1)
                # print('suffix_constraints 2:', suffix_constraints, flush=True)
                src_lengths = src_lengths[batch_idxs]
                # cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, scores.shape[-1])
                # scores_full = scores_full.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, tokens.shape[-1])
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
                active_bsz_idx = active_bsz_idx.masked_select(batch_mask)
                active_hypos_cand_idx = torch.arange(0, beam_size).repeat(bsz, 1).type_as(tokens).to(
                    src_tokens.device).add(cand_bbsz_offsets).view(-1)
            else:
                batch_idxs = None
            if active_bsz_idx.numel() == 0 or current_max_step>=max_len:
                for sent, hp in enumerate(finalized_heap):
                    while len(hp) > 0:
                        finalized[sent].insert(0, heapq.heappop(hp)[-1])
                break
            active_bbsz_idx = cand_bbsz_idx[:, :beam_size]
            active_scores = cand_scores[:, :beam_size]

            active_bbsz_idx = active_bbsz_idx.reshape(-1)
            active_scores = active_scores.reshape(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            # print('tokens before: ', tokens[:, :current_max_step+1], flush=True)
            # print('active_bbsz_idx: ', active_bbsz_idx, flush=True)
            tokens[:, : current_max_step + 1] = torch.index_select(
                tokens[:, : current_max_step + 1], dim=0, index=active_bbsz_idx
            )
            # print('tokens after: ', tokens[:, :current_max_step+1], flush=True)
            scores[:, : current_max_step] = torch.index_select(
                scores[:, : current_max_step], dim=0, index=active_bbsz_idx
            )
            # scores_full[:, : current_max_step] = torch.index_select(
            #     scores_full[:, :current_max_step], dim=0, index=active_bbsz_idx
            # )
            # Select the next token for each of them
            # print('output_tokens before:', output_tokens[:,:current_max_step], flush=True)
            # print('active_hypos_cand_idx:', active_hypos_cand_idx, flush=True)
            # output_tokens = output_tokens[active_hypos_cand_idx]
            # print('output_tokens after:', output_tokens[:,:current_max_step], flush=True)
            cand_steps = cand_steps[active_hypos_cand_idx]
            cand_start_steps = cand_start_steps[active_hypos_cand_idx]
            output_tokens = output_tokens[active_hypos_cand_idx]
            output_tokens_index = torch.arange(output_tokens.size(-1)).to(output_tokens).repeat(bsz * beam_size, 1)
            output_tokens_mask = output_tokens_index.ge(cand_steps.unsqueeze(-1))
            output_tokens[output_tokens_mask] = self.pad
            tokens[:, 1:output_tokens.size(1) + 1] = output_tokens[:,:max_len+1]
            # print('tokens after 2: ', tokens[:,:current_max_step+1], flush=True)
            # cand_scores_next_step_mask = cand_scores_next_step_mask[active_hypos_cand_idx]
            # cand_scores_avg = cand_scores_avg[active_hypos_cand_idx]
            # scores_full[cand_scores_next_step_mask] = cand_scores_avg
            # Update constraints based on which candidates were selected for the next beam
            # self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : current_max_len + 1] = torch.index_select(
                    attn[:, :, : current_max_len + 1], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx
            steps = cand_steps + 1
            start_steps = cand_start_steps
        torch.cuda.empty_cache()
        if return_time_ignore:
            return finalized, times_to_ignore
        else:
            return finalized



        #     # finalize hypotheses spans by early stop
        #     cand_scores_full_indices = scores_full_indices[cand_bbsz_idx.view(-1)]
        #     cand_scores_full_mask = torch.logical_or(cand_scores_full_indices.lt(cand_start_steps.unsqueeze(-1)),
        #                                          cand_scores_full_indices.ge(cand_steps.unsqueeze(-1)))
        #     cand_scores_next_step_mask = cand_scores_full_indices.eq((cand_steps-1).unsqueeze(-1))
        #     cand_scores_full = scores_full[cand_bbsz_idx.view(-1)]
        #     cand_scores_full[cand_scores_next_step_mask] = cand_scores_avg
        #     cand_scores_full_cache = torch.zeros_like(cand_scores_full).copy_(cand_scores_full).to(cand_scores_full)
        #     cand_scores_full[cand_scores_full_mask] = -math.inf
        #     # print('cand_scores_full: ', cand_scores_full, flush=True)
        #     span_end_scores, span_end_step = cand_scores_full.max(dim=1)
        #     # print('span_end_step: ', span_end_step, flush=True)
        #     span_end_mask = (cand_steps - span_end_step).gt(patience).view(bsz, cand_size)
        #     # print('span_end_mask: ', span_end_mask, flush=True)
        #     span_end_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(span_end_mask)
        #     span_end_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=span_end_mask[:, :beam_size])
        #     end_beam_num = span_end_bbsz_idx.numel()
        #     finalized_sents: List[int] = []
        #     if end_beam_num > 0:
        #         span_end_step = torch.masked_select(span_end_step.view(bsz, cand_size)[:, :beam_size], mask=span_end_mask[:, :beam_size])
        #         span_step = torch.masked_select(cand_steps.view(bsz, cand_size)[:, :beam_size], mask=span_end_mask[:, :beam_size])
        #         tokens_clone = torch.masked_select(output_tokens.view(bsz, cand_size, -1)[:, :beam_size, :], mask=span_end_mask[:, :beam_size].unsqueeze(-1)).view(end_beam_num, current_max_len)
        #         # print('tokens_clone: ', tokens_clone, flush=True)
        #         tokens_clone_len = torch.masked_select((tokens_with_suffix_constraints_real_lens-1).view(bsz, cand_size)[:, :beam_size], mask=span_end_mask[:, :beam_size])
        #         tokens_index = torch.arange(tokens_clone.size(-1)).to(tokens).repeat(end_beam_num, 1)
        #         extra_tokens_mask = torch.logical_and(tokens_index.gt(span_end_step.unsqueeze(-1)), tokens_index.lt(span_step.unsqueeze(-1)))# mask extra decoded tokens after end step and before suffix tokens
        #         tokens_clone_masked = torch.zeros_like(tokens_clone).copy_(tokens_clone).to(tokens_clone)
        #         tokens_clone_masked[extra_tokens_mask] = self.pad
        #         tokens_mask = tokens_clone_masked.ne(self.pad)
        #         tokens_length = tokens_mask.sum(dim=-1)
        #         tokens_strip = torch.split(tokens_clone_masked[tokens_mask], tokens_length.tolist())
        #         # print('tokens_strip: ', tokens_strip, flush=True)
        #         cand_scores_full_clone = torch.masked_select(cand_scores_full_cache.view(bsz, cand_size, -1)[:, :beam_size, :], mask=span_end_mask[:, :beam_size].unsqueeze(-1)).view(end_beam_num, -1)
        #
        #         eos_scores = torch.masked_select(span_end_scores.view(bsz, cand_size)[:, :beam_size], mask=span_end_mask[:, :beam_size]).view(end_beam_num, -1).squeeze(-1)
        #         attn_clone = torch.masked_select(avg_attn_scores_with_suffix.view(bsz, cand_size, -1)[:, :beam_size, :], mask=span_end_mask[:, :beam_size].unsqueeze(-1)).view(end_beam_num, -1)
        #
        #         cum_unfin: List[int] = []
        #         prev = 0
        #         for f in finished:
        #             if f:
        #                 prev += 1
        #             else:
        #                 cum_unfin.append(prev)
        #         cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(span_end_bbsz_idx)
        #         unfin_idx = torch.div(span_end_bbsz_idx, beam_size, rounding_mode="trunc")
        #         sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)
        #         seen = (sent << 32) + unfin_idx
        #         unique_seen: List[int] = torch.unique(seen).tolist()
        #
        #         sent_list: List[int] = sent.tolist()
        #         for i in range(end_beam_num):
        #             if len(finalized[sent_list[i]]) < beam_size:
        #                 if attn_clone is not None:
        #                     hypo_attn = attn_clone[i]
        #                 else:
        #                     hypo_attn = torch.empty(0)
        #
        #                 finalized[sent_list[i]].append({
        #                     "tokens": tokens_strip[i],
        #                     "concat_tokens": tokens_clone[i],
        #                     "score": eos_scores[i],
        #                     "attention": hypo_attn,
        #                     "alignment": torch.empty(0),
        #                     "positional_scores": cand_scores_full_clone[i][:span_step[i]+1],
        #                     "scores": scores.index_select(0, span_end_bbsz_idx)[i, :tokens_clone_len[i]]
        #                 })
        #                 # print('i:', i, 'sent_list[i]:', sent_list[i], 'finalized[sent_list[i]]:', finalized[sent_list[i]], flush=True)
        #
        #         for unique_s in unique_seen:
        #             unique_sent: int = unique_s >> 32
        #             unique_unfin_idx: int = unique_s - (unique_sent << 32)
        #             if not finished[unique_sent] and self.is_finished(
        #                     0, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
        #             ): # Todo: consider max_len
        #                 finished[unique_sent] = True
        #                 finalized_sents.append(unique_unfin_idx)
        #
        #         num_remaining_sent -= len(finalized_sents)
        #
        #     assert num_remaining_sent >= 0
        #     if num_remaining_sent == 0:
        #         break
        #     eos_mask = span_end_mask
        #     # Remove finalized sentences (ones for which {beam_size}
        #     # finished hypotheses have been generated) from the batch.
        #     if len(finalized_sents) > 0:
        #         new_bsz = bsz - len(finalized_sents)
        #
        #         # construct batch_idxs which holds indices of batches to keep for the next pass
        #         batch_mask = torch.ones(
        #             bsz, dtype=torch.bool, device=cand_indices.device
        #         )
        #         batch_mask[finalized_sents] = False
        #         # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
        #         batch_idxs = torch.arange(
        #             bsz, device=cand_indices.device
        #         ).masked_select(batch_mask)
        #
        #         # Choose the subset of the hypothesized constraints that will continue
        #         self.search.prune_sentences(batch_idxs)
        #
        #         eos_mask = span_end_mask[batch_idxs]
        #         cand_beams = cand_beams[batch_idxs]
        #         bbsz_offsets.resize_(new_bsz, 1)
        #         cand_bbsz_offsets.resize_(new_bsz, 1)
        #         cand_bbsz_idx = cand_beams.add(bbsz_offsets)
        #         cand_scores = cand_scores[batch_idxs]
        #         cand_indices = cand_indices[batch_idxs]
        #         span_end_scores = span_end_scores.view(bsz, cand_size)[batch_idxs].view(-1)
        #         cand_scores_avg = cand_scores_avg.view(bsz, cand_size)[batch_idxs].view(-1)
        #         cand_scores_next_step_mask = cand_scores_next_step_mask.view(bsz, cand_size, -1)[batch_idxs].view(new_bsz*cand_size, -1)
        #         cand_steps = cand_steps.view(bsz, cand_size)[batch_idxs].view(-1)
        #         cand_start_steps = cand_start_steps.view(bsz, cand_size)[batch_idxs].view(-1)
        #         # tokens_with_suffix_constraints_real_lens = tokens_with_suffix_constraints_real_lens.view(bsz, cand_size)[batch_idxs].view(-1)
        #         output_tokens = output_tokens.view(bsz, cand_size, -1)[batch_idxs].view(new_bsz*cand_size, -1)
        #         # cand_start_steps = cand_start_steps.view(bsz, cand_size)[batch_idxs].view(-1)
        #         # cand_suffix_constraints = [list(itertools.chain(*x, [self.eos])) if len(x) > 0 else [self.eos] for x in
        #         #                            suffix_constraints[cand_bbsz_idx.view(-1)]]
        #         # cand_suffix_lens = torch.Tensor([len(x) for x in cand_suffix_constraints]).to(steps)
        #         # cand_suffix_constraints_padded = torch.nn.utils.rnn.pad_sequence(
        #         #     [torch.Tensor(x) for x in cand_suffix_constraints], batch_first=True, padding_value=self.pad).to(
        #         #     tokens)
        #         # tokens_with_suffix_constraints = torch.cat(
        #         #     [tokens[cand_bbsz_idx.view(-1), :current_max_step + 1], cand_indices.view(-1).unsqueeze(-1),
        #         #      cand_suffix_constraints_padded], dim=-1)
        #         # tokens_with_suffix_constraints_real_lens = cand_steps + 2 + cand_suffix_lens  # +2 for bos and eos
        #         # tokens_with_suffix_constraints = torch.nn.utils.rnn.pad_sequence(
        #         #     torch.split(tokens_with_suffix_constraints[tokens_with_suffix_constraints.ne(self.pad)],
        #         #                 tokens_with_suffix_constraints_real_lens.tolist()), batch_first=True,
        #         #     padding_value=self.pad).to(tokens)
        #         batch_idxs_cpu = batch_idxs.cpu()
        #         prefix_constraints = prefix_constraints.reshape(bsz, beam_size)[batch_idxs_cpu].reshape(-1)
        #         # print('suffix_constraints 1:', suffix_constraints, flush=True)
        #         # print('batch_idxs_cpu: ', batch_idxs_cpu)
        #         suffix_constraints = suffix_constraints.reshape(bsz, beam_size)[batch_idxs_cpu].reshape(-1)
        #         # print('suffix_constraints 2:', suffix_constraints, flush=True)
        #         src_lengths = src_lengths[batch_idxs]
        #         cands_to_ignore = cands_to_ignore[batch_idxs]
        #
        #         scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
        #         scores_full = scores_full.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
        #         tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
        #         if attn is not None:
        #             attn = attn.view(bsz, -1)[batch_idxs].view(
        #                 new_bsz * beam_size, attn.size(1), -1
        #             )
        #         bsz = new_bsz
        #     else:
        #         batch_idxs = None
        #
        #     # Set active_mask so that values > cand_size indicate eos hypos
        #     # and values < cand_size indicate candidate active hypos.
        #     # After, the min values per row are the top candidate active hypos
        #
        #     # Rewrite the operator since the element wise or is not supported in torchscript.
        #
        #     eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
        #     eos_mask[:, beam_size:] = False
        #     active_mask = torch.add(
        #         eos_mask.type_as(cand_offsets) * cand_size,
        #         cand_offsets[: eos_mask.size(1)],
        #     )
        #     overall_scores = span_end_scores.view(bsz, cand_size)
        #     overall_scores[eos_mask] = torch.tensor(-math.inf).to(overall_scores)
        #     _, active_hypos = torch.topk(
        #         overall_scores, k=beam_size, dim=1
        #     )
        #     new_cands_to_ignore = active_mask.view(-1)[active_hypos.add(cand_bbsz_offsets).view(-1)].view(bsz, beam_size)
        #
        #     # get the top beam_size active hypotheses, which are just
        #     # the hypos with the smallest values in active_mask.
        #     # {active_hypos} indicates which {beam_size} hypotheses
        #     # from the list of {2 * beam_size} candidates were
        #     # selected. Shapes: (batch size, beam size)
        #     # new_cands_to_ignore, active_hypos = torch.topk(
        #     #     active_mask, k=beam_size, dim=1, largest=False
        #     # )
        #
        #     # update cands_to_ignore to ignore any finalized hypos.
        #     cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
        #     # Make sure there is at least one active item for each sentence in the batch.
        #     assert (~cands_to_ignore).any(dim=1).all()
        #
        #     # update cands_to_ignore to ignore any finalized hypos
        #
        #     # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
        #     # can be selected more than once).
        #     active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
        #     active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)
        #     active_hypos_cand_idx = active_hypos.add(cand_bbsz_offsets).view(-1)
        #
        #     active_bbsz_idx = active_bbsz_idx.view(-1)
        #     active_scores = active_scores.view(-1)
        #
        #     # copy tokens and scores for active hypotheses
        #
        #     # Set the tokens for each beam (can select the same row more than once)
        #     # print('tokens before: ', tokens[:, :current_max_step+1], flush=True)
        #     # print('active_bbsz_idx: ', active_bbsz_idx, flush=True)
        #     tokens[:, : current_max_step+1] = torch.index_select(
        #         tokens[:, : current_max_step+1], dim=0, index=active_bbsz_idx
        #     )
        #     # print('tokens after: ', tokens[:, :current_max_step+1], flush=True)
        #     scores[:, : current_max_step] = torch.index_select(
        #         scores[:, : current_max_step], dim=0, index=active_bbsz_idx
        #     )
        #     scores_full[:, : current_max_step] = torch.index_select(
        #         scores_full[:, :current_max_step], dim=0, index=active_bbsz_idx
        #     )
        #     # Select the next token for each of them
        #     # print('output_tokens before:', output_tokens[:,:current_max_step], flush=True)
        #     # print('active_hypos_cand_idx:', active_hypos_cand_idx, flush=True)
        #     output_tokens = output_tokens[active_hypos_cand_idx]
        #     # print('output_tokens after:', output_tokens[:,:current_max_step], flush=True)
        #     cand_steps = cand_steps[active_hypos_cand_idx]
        #     cand_start_steps = cand_start_steps[active_hypos_cand_idx]
        #     output_tokens_index = torch.arange(output_tokens.size(-1)).to(output_tokens).repeat(bsz*beam_size, 1)
        #     output_tokens_mask = output_tokens_index.ge(cand_steps.unsqueeze(-1))
        #     output_tokens[output_tokens_mask] = self.pad
        #     tokens[:, 1:output_tokens.size(1)+1] = output_tokens
        #     # print('tokens after 2: ', tokens[:,:current_max_step+1], flush=True)
        #     cand_scores_next_step_mask = cand_scores_next_step_mask[active_hypos_cand_idx]
        #     cand_scores_avg = cand_scores_avg[active_hypos_cand_idx]
        #     scores_full[cand_scores_next_step_mask] = cand_scores_avg
        #     # Update constraints based on which candidates were selected for the next beam
        #     self.search.update_constraints(active_hypos)
        #
        #     # copy attention for active hypotheses
        #     if attn is not None:
        #         attn[:, :, : current_max_len + 1] = torch.index_select(
        #             attn[:, :, : current_max_len + 1], dim=0, index=active_bbsz_idx
        #         )
        #
        #     # reorder incremental state in decoder
        #     reorder_state = active_bbsz_idx
        #     steps = cand_steps + 1
        #     start_steps = cand_start_steps
        #
        # # sort by score descending
        # for sent in range(len(finalized)):
        #     scores = torch.tensor(
        #         [float(elem["score"].item()) for elem in finalized[sent]]
        #     )
        #     _, sorted_scores_indices = torch.sort(scores, descending=True)
        #     finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
        #     finalized[sent] = torch.jit.annotate(
        #         List[Dict[str, Tensor]], finalized[sent]
        #     )
        # if return_time_ignore:
        #     return finalized, times_to_ignore
        # else:
        #     return finalized

    def _generate_with_suffix_early_stop(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
            suffix_tokens: Optional[Tensor] = None,
            patience=10
    ):

        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception(
                "expected src_tokens or source in net input. input keys: "
                + str(net_input.keys())
            )

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
            encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        scores_full = torch.zeros(bsz * beam_size, max_len + 1).to(scores)
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(src_tokens)
                .long()
                .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # eos_lprobs = (
        #     torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        # )
        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
                .unsqueeze(1)
                .type_as(tokens)
                .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            with torch.autograd.profiler.record_function(
                    "EnsembleModel: forward_decoder"
            ):
                lprobs, avg_attn_scores = self.model.forward_decoder(
                    tokens[:, : step + 1],
                    encoder_outs,
                    incremental_states,
                    self.temperature,
                )

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty
            lprobs[:, self.eos] = -math.inf  # never select eos when we have suffix

            # Todo: handl max length when has suffix
            # handle max length constraint
            # if step >= max_len:
            #     lprobs[:, : self.eos] = -math.inf
            #     lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                    prefix_tokens is not None
                    and step < prefix_tokens.size(1)
                    and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            # Todo: consider min len when has suffix
            # elif step < self.min_len:
            #     # minimum length constraint (does not apply if using prefix_tokens)
            #     lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            # if avg_attn_scores is not None:
            #     if attn is None:
            #         attn = torch.empty(
            #             bsz * beam_size, avg_attn_scores.size(1), max_len + 2
            #         ).to(scores)
            #     attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores_ori, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )
            # cand_eos_lprobs = lprobs.view(bsz, -1)[:, cand_beams * self.vocab_size + self.eos][:, 0, :]

            # forward with suffix
            # cand_scores_partial, cand_indices, cand_beams = cand_scores[:, :beam_size], cand_indices[:, :beam_size], cand_beams[:, :beam_size]
            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            suffix_toks = suffix_tokens.unsqueeze(-1).repeat(1, 1, cand_size).transpose(1, 2).reshape(bsz * cand_size,
                                                                                                      -1)
            # Todo: consider suffix_tokens have different lengths
            tokens_with_suffix = torch.cat(
                [tokens[cand_bbsz_idx.view(-1), :step + 1], cand_indices.view(-1, 1), suffix_toks], dim=-1)

            with torch.autograd.profiler.record_function(
                    "EnsembleModel: forward_decoder"
            ):
                lprobs_with_suffix, avg_attn_scores_with_suffix = self.model.forward_decoder(
                    tokens_with_suffix,
                    encoder_outs,
                    None,
                    self.temperature,
                    return_all_prob=True
                )
                # print('avg_attn_scores_with_suffix.shape: ', avg_attn_scores_with_suffix.size(), flush=True)
                avg_attn_scores_with_suffix = avg_attn_scores_with_suffix.view(bsz, cand_size, -1)[:, :beam_size,
                                              :].view(bsz * beam_size, -1)

            if self.lm_model is not None:
                lm_out_with_suffix = self.lm_model(tokens_with_suffix)
                probs_with_suffix = self.lm_model.get_normalized_probs(
                    lm_out_with_suffix, log_probs=True, sample=None
                )
                probs_with_suffix = probs_with_suffix * self.lm_weight  # probs_with_suffix[:, -1, :] * self.lm_weight
                lprobs_with_suffix += probs_with_suffix

            if avg_attn_scores_with_suffix is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores_with_suffix.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores_with_suffix)

            prob_index = torch.cat([tokens_with_suffix[:, 1:],
                                    self.eos * (torch.ones(tokens_with_suffix.size(0))).to(
                                        tokens_with_suffix).unsqueeze(-1)], dim=-1)
            cand_scores = lprobs_with_suffix.gather(-1, prob_index.unsqueeze(-1)).sum(dim=1).squeeze(-1).view(bsz,
                                                                                                              cand_size)
            # finalize hypotheses that overall score has no increase for `patience` steps
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = torch.zeros(bsz, cand_size).to(tokens.device).bool()
            eos_step = torch.zeros(bsz, cand_size).to(tokens)
            start_step = 1 if prefix_tokens is None else (prefix_tokens.size(1) + 1)  # extra 1 for bos
            # print('cand_bbsz_idx: ', cand_bbsz_idx.shape, cand_bbsz_idx, flush=True)
            # print('cand_scores: ', cand_scores.shape, cand_scores, flush=True)
            cand_scores_seq = torch.cat(
                [scores_full[cand_bbsz_idx.view(-1), start_step - 1:step], cand_scores.view(-1).unsqueeze(-1)],
                dim=-1).view(bsz,
                             cand_size,
                             -1)

            # Todo: consider prefix with different lengths, i.e. padding in prefix_tokens
            if step >= start_step + patience:
                cand_scores_seq_avg = cand_scores_seq.div(
                    (torch.arange(start_step - 1, step + 1) + 1 + suffix_tokens.size(1) + 1).to(cand_scores_seq))
                max_score, eos_step = torch.max(cand_scores_seq_avg, dim=-1)
                eos_step = (eos_step + start_step - 1)  # The step that eos score reaches highest
                eos_mask = (step - eos_step).ge(patience)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size])

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                end_step = torch.masked_select(eos_step[:, :beam_size], mask=eos_mask[:, :beam_size])  # [eos_bbsz]
                cand_scores_seq = torch.masked_select(
                    cand_scores_seq[:, :beam_size], mask=eos_mask[:, :beam_size].unsqueeze(-1)
                ).view(-1, step - start_step + 2)  # [eos_bbsz, step+2-start_step]
                tokens_clone = torch.cat(
                    [tokens[cand_bbsz_idx.view(-1), start_step:step + 1], cand_indices.view(-1, 1)],
                    dim=-1).index_select(0, eos_bbsz_idx)
                prefix_tokens_clone = None if prefix_tokens is None else \
                    prefix_tokens.unsqueeze(-1).repeat(1, 1, cand_size).transpose(1, 2).reshape(bsz * cand_size, -1)[
                        cand_bbsz_idx.view(-1)].index_select(0, eos_bbsz_idx)
                suffix_tokens_clone = None if suffix_tokens is None else \
                    suffix_tokens.unsqueeze(-1).repeat(1, 1, cand_size).transpose(1, 2).reshape(bsz * cand_size, -1)[
                        cand_bbsz_idx.view(-1)].index_select(0, eos_bbsz_idx)
                finalized_sents = self.finalize_hypos_with_suffix_patience(
                    step,
                    start_step,
                    end_step,
                    eos_bbsz_idx,
                    cand_scores_seq,
                    tokens_clone,
                    prefix_tokens_clone,
                    suffix_tokens_clone,
                    scores,
                    scores_full,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                    # eos_lprobs
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_scores_ori = cand_scores_ori[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                if suffix_tokens is not None:
                    suffix_tokens = suffix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_full = scores_full.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)

                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            # print('cand_scores: ', cand_scores.shape, cand_scores, flush=True)
            # print('active_hypos: ', active_hypos.shape, active_hypos, flush=True)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
                scores_full[:, :step] = torch.index_select(
                    scores_full[:, :step], dim=0, index=active_bbsz_idx
                )

            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores_ori, dim=1, index=active_hypos
            )
            scores_full.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:beam_size, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def _generate_with_suffix(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
            suffix_tokens: Optional[Tensor] = None
    ):

        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception(
                "expected src_tokens or source in net input. input keys: "
                + str(net_input.keys())
            )

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
            encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(src_tokens)
                .long()
                .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # eos_lprobs = (
        #     torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        # )
        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
                .unsqueeze(1)
                .type_as(tokens)
                .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            with torch.autograd.profiler.record_function(
                    "EnsembleModel: forward_decoder"
            ):
                lprobs, avg_attn_scores = self.model.forward_decoder(
                    tokens[:, : step + 1],
                    encoder_outs,
                    incremental_states,
                    self.temperature,
                )

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty
            lprobs[:, self.eos] = -math.inf  # never select eos when we have suffix

            # Todo: handl max length when has suffix
            # handle max length constraint
            # if step >= max_len:
            #     lprobs[:, : self.eos] = -math.inf
            #     lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                    prefix_tokens is not None
                    and step < prefix_tokens.size(1)
                    and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            # Todo: consider min len when has suffix
            # elif step < self.min_len:
            #     # minimum length constraint (does not apply if using prefix_tokens)
            #     lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            # if avg_attn_scores is not None:
            #     if attn is None:
            #         attn = torch.empty(
            #             bsz * beam_size, avg_attn_scores.size(1), max_len + 2
            #         ).to(scores)
            #     attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )
            # cand_eos_lprobs = lprobs.view(bsz, -1)[:, cand_beams * self.vocab_size + self.eos][:, 0, :]

            # forward with suffix
            # cand_scores_partial, cand_indices, cand_beams = cand_scores[:, :beam_size], cand_indices[:, :beam_size], cand_beams[:, :beam_size]
            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            suffix_toks = suffix_tokens.unsqueeze(-1).repeat(1, 1, cand_size).transpose(1, 2).reshape(bsz * cand_size,
                                                                                                      -1)
            # Todo: consider suffix_tokens have different lengths
            tokens_with_suffix = torch.cat(
                [tokens[cand_bbsz_idx.view(-1), :step + 1], cand_indices.view(-1, 1), suffix_toks], dim=-1)

            with torch.autograd.profiler.record_function(
                    "EnsembleModel: forward_decoder"
            ):
                lprobs_with_suffix, avg_attn_scores_with_suffix = self.model.forward_decoder(
                    tokens_with_suffix,
                    encoder_outs,
                    None,
                    self.temperature,
                )
                # print('avg_attn_scores_with_suffix.shape: ', avg_attn_scores_with_suffix.size(), flush=True)
                avg_attn_scores_with_suffix = avg_attn_scores_with_suffix.view(bsz, cand_size, -1)[:, :beam_size,
                                              :].view(bsz * beam_size, -1)

            if self.lm_model is not None:
                lm_out_with_suffix = self.lm_model(tokens_with_suffix)
                probs_with_suffix = self.lm_model.get_normalized_probs(
                    lm_out_with_suffix, log_probs=True, sample=None
                )
                probs_with_suffix = probs_with_suffix[:, -1, :] * self.lm_weight
                lprobs_with_suffix += probs_with_suffix
            if avg_attn_scores_with_suffix is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores_with_suffix.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores_with_suffix)
            cand_scores = lprobs_with_suffix[:, self.eos].view(bsz, cand_size)
            _, last_token = torch.max(lprobs_with_suffix.view(bsz, cand_size, -1), dim=-1)
            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = last_token.eq(self.eos)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)
            if prefix_tokens is not None and step < prefix_tokens.size(1):
                eos_mask[:] = False
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size])

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos_with_suffix(
                    step + 1 + suffix_toks.size(1),
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    torch.cat([tokens_with_suffix,
                               torch.zeros(tokens_with_suffix.size(0)).to(tokens_with_suffix).unsqueeze(-1)], dim=-1),
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                    # eos_lprobs
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                if suffix_tokens is not None:
                    suffix_tokens = suffix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)

                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )

            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:beam_size, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
            suffix_tokens: Optional[int] = None,
            patience: Optional[int] = 0,
            return_time_ignore = False,
            **kwargs
    ):
        if patience > 0:
            assert constraints is None or type(constraints) in (tuple, list)
            return self._generate_with_constraints_early_stop(sample, constraints, bos_token, patience, return_time_ignore=return_time_ignore, **kwargs)
        if suffix_tokens is not None:
            return self._generate_with_suffix_early_stop(sample, prefix_tokens, constraints, bos_token, suffix_tokens)
            # return self._generate_with_suffix(sample, prefix_tokens, constraints, bos_token, suffix_tokens)
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception(
                "expected src_tokens or source in net input. input keys: "
                + str(net_input.keys())
            )

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
            encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(src_tokens)
                .long()
                .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        eos_lprobs = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )
        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
                .unsqueeze(1)
                .type_as(tokens)
                .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            with torch.autograd.profiler.record_function(
                    "EnsembleModel: forward_decoder"
            ):
                lprobs, avg_attn_scores = self.model.forward_decoder(
                    tokens[:, : step + 1],
                    encoder_outs,
                    None, # incremental_states,
                    self.temperature,
                )

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                    prefix_tokens is not None
                    and step < prefix_tokens.size(1)
                    and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )
            cand_eos_lprobs = lprobs.reshape(bsz, -1)[:, cand_beams * self.vocab_size + self.eos][:, 0, :]

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                    eos_lprobs
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                cand_eos_lprobs = cand_eos_lprobs[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                eos_lprobs = eos_lprobs.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
                eos_lprobs[:, : step] = torch.index_select(
                    eos_lprobs[:, : step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )
            eos_lprobs.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_eos_lprobs, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        if return_time_ignore:
            return finalized, 0.0
        else:
            return finalized

    def _prefix_tokens(
            self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                         :, 0, 1: step + 1
                         ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def _suffix_tokens(
            self, step: int, tokens, suffix_tokens, beam_size, encoder_outs: Tensor, incremental_states: Tensor
    ):
        """Handle suffix tokens"""
        # tokens: [bbsz, L]
        # suffix_tokens: [bbsz, L]]
        bsz = suffix_tokens.shape[0]
        suffix_toks = suffix_tokens.unsqueeze(-1).repeat(1, 1, beam_size).squeeze().transpose(1, 2).reshape(
            bsz * beam_size, -1)
        # Todo: consider suffix_tokens have different lengths
        input_toks = torch.cat([tokens[:, :step], suffix_toks], dim=-1)

        with torch.autograd.profiler.record_function(
                "EnsembleModel: forward_decoder"
        ):
            lprobs, _ = self.model.forward_decoder(
                input_toks,
                encoder_outs,
                incremental_states,
                self.temperature,
            )

        if self.lm_model is not None:
            lm_out = self.lm_model(input_toks)
            probs = self.lm_model.get_normalized_probs(
                lm_out, log_probs=True, sample=None
            )
            probs = probs[:, -1, :] * self.lm_weight
            lprobs += probs
        suffix_scores = lprobs[:, self.eos]
        return lprobs, input_toks, suffix_scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos_with_suffix_patience(
            self,
            step: int,
            start_step: int,
            end_step: Tensor,
            bbsz_idx,
            cand_scores_seq: Tensor,
            tokens_clone: Tensor,
            prefix_tokens_clone: Optional[Tensor],
            suffix_tokens_clone: Optional[Tensor],
            scores,
            scores_full,
            finalized: List[List[Dict[str, Tensor]]],
            finished: List[bool],
            beam_size: int,
            attn: Optional[Tensor],
            src_lengths,
            max_len: int
    ):
        # print(f'step: {step}, start_step: {start_step}, end_step: {end_step}, bbsz_idx: {bbsz_idx}', flush=True)
        # print('cand_scores_seq: ', cand_scores_seq, flush=True)
        # print('tokens_clone: ', tokens_clone, flush=True)
        # print('prefix_tokens_clone: ', prefix_tokens_clone, flush=True)
        # print('suffix_tokens_clone: ', suffix_tokens_clone, flush=True)
        # print('scores: ', scores)
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        eos_scores = cand_scores_seq.gather(-1, end_step.unsqueeze(-1) - start_step + 1).squeeze(-1)
        assert bbsz_idx.numel() == eos_scores.numel()
        tokens_to_cat = []
        if prefix_tokens_clone is not None:
            tokens_to_cat.append(prefix_tokens_clone)
        tokens_to_cat.append(tokens_clone)
        if suffix_tokens_clone is not None:
            tokens_to_cat.append(suffix_tokens_clone)
        eos_pad = self.eos * (torch.ones(tokens_clone.size(0)).long()).to(tokens_clone)
        tokens_to_cat.append(eos_pad.unsqueeze(-1))
        concat_tokens = torch.cat(tokens_to_cat, dim=-1)  # [:, 1:]  # remove bos
        bbsz, m_len = concat_tokens.size()
        real_len = m_len - step + end_step
        index_matrix = torch.arange(m_len).repeat((bbsz, 1)).to(end_step)
        tokens_mask = torch.logical_or(index_matrix <= end_step.unsqueeze(-1), index_matrix > step)
        tokens_strip = torch.split(torch.masked_select(concat_tokens, mask=tokens_mask), real_len.tolist())
        # tokens_strip = torch.nn.utils.rnn.pad_sequence(tokens_strip, batch_first=True, padding_value=self.pad)
        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}

        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1: step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = cand_scores_seq[:, step - start_step]
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]
        pos_scores_full = scores_full.index_select(0, bbsz_idx)[:, :step + 1]
        pos_scores_full[:, step] = cand_scores_seq[:, step - start_step]
        pos_scores_full[:, 1:] = pos_scores_full[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores = eos_scores.div(real_len ** self.len_penalty)
            # eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = torch.div(bbsz_idx, beam_size, rounding_mode="trunc")
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        if self.match_source_len:
            condition = step > torch.index_select(src_lengths, 0, unfin_idx)
            eos_scores = torch.where(condition, torch.tensor(-math.inf), eos_scores)
        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent_list[i]].append(
                    {
                        "tokens": tokens_strip[i],
                        "concat_tokens": concat_tokens[i],
                        "score": eos_scores[i],
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                        "positional_scores_full": pos_scores_full[i],
                        "scores": scores.index_select(0, bbsz_idx)[i, : step + 1],
                        "scores_full": scores_full.index_select(0, bbsz_idx)[i, : step + 1]
                    }
                )

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                    step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def finalize_hypos_with_suffix(
            self,
            step: int,
            no_suffix_step: int,
            bbsz_idx,
            eos_scores,
            tokens,
            scores,
            finalized: List[List[Dict[str, Tensor]]],
            finished: List[bool],
            beam_size: int,
            attn: Optional[Tensor],
            src_lengths,
            max_len: int
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
                       :, 1: step + 2
                       ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1: step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : no_suffix_step + 1]
        pos_scores[:, no_suffix_step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = torch.div(bbsz_idx, beam_size, rounding_mode="trunc")
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        if self.match_source_len:
            condition = step > torch.index_select(src_lengths, 0, unfin_idx)
            eos_scores = torch.where(condition, torch.tensor(-math.inf), eos_scores)
        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent_list[i]].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": eos_scores[i],
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                        "scores": scores.index_select(0, bbsz_idx)[i, : no_suffix_step + 1]
                    }
                )

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                    step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def finalize_hypos(
            self,
            step: int,
            bbsz_idx,
            eos_scores,
            tokens,
            scores,
            finalized: List[List[Dict[str, Tensor]]],
            finished: List[bool],
            beam_size: int,
            attn: Optional[Tensor],
            src_lengths,
            max_len: int,
            eos_lprobs: Optional[Tensor]
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
                       :, 1: step + 2
                       ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1: step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        pos_eos_lprobs = eos_lprobs.index_select(0, bbsz_idx)[:, : step + 1]
        pos_eos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_eos_scores[:, 1:] = pos_eos_scores[:, :-1] + pos_eos_lprobs[:, 1:]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = torch.div(bbsz_idx, beam_size, rounding_mode="trunc")
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        if self.match_source_len:
            condition = step > torch.index_select(src_lengths, 0, unfin_idx)
            eos_scores = torch.where(condition, torch.tensor(-math.inf), eos_scores)
        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent_list[i]].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": eos_scores[i],
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                        "scores": scores.index_select(0, bbsz_idx)[i, : step + 1],
                        "pos_eos_lprobs": pos_eos_lprobs[i],
                        "pos_eos_scores": pos_eos_scores[i]
                    }
                )

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                    step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def is_finished(
            self,
            step: int,
            unfin_idx: int,
            max_len: int,
            finalized_sent_len: int,
            beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.has_incremental: bool = False
        if all(
                hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
                for m in models
        ):
            self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min(
            [
                m.max_decoder_positions()
                for m in self.models
                if hasattr(m, "max_decoder_positions")
            ]
            + [sys.maxsize]
        )

    def set_decoder_beam_size(self, beam_size):
        """Set beam size for efficient beamable enc-dec attention."""
        if beam_size > 1:
            for model in self.models:
                if hasattr(model, "set_beam_size"):
                    model.set_beam_size(beam_size)

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(
            self,
            tokens,
            encoder_outs: List[Dict[str, List[Tensor]]],
            incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
            temperature: float = 1.0,
            return_all_prob: bool = False
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states() and incremental_states is not None:
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
            else:
                if hasattr(model, "decoder"):
                    decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
                else:
                    decoder_out = model.forward(tokens)

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0].div_(temperature),  # decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )
            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            if not return_all_prob:
                probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_encoder_out(
            self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order
    ):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
            self,
            incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
            new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )


class SequenceGeneratorWithAlignment(SequenceGenerator):
    def __init__(
            self, models, tgt_dict, left_pad_target=False, print_alignment="hard", **kwargs
    ):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(EnsembleModelWithAlignment(models), tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

        if print_alignment == "hard":
            self.extract_alignment = utils.extract_hard_alignment
        elif print_alignment == "soft":
            self.extract_alignment = utils.extract_soft_alignment

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        finalized = super()._generate(sample, **kwargs)

        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        (
            src_tokens,
            src_lengths,
            prev_output_tokens,
            tgt_tokens,
        ) = self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, "full_context_alignment", False) for m in self.model.models):
            attn = self.model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [
                finalized[i // beam_size][i % beam_size]["attention"].transpose(1, 0)
                for i in range(bsz * beam_size)
            ]

        if src_tokens.device != "cpu":
            src_tokens = src_tokens.to("cpu")
            tgt_tokens = tgt_tokens.to("cpu")
            attn = [i.to("cpu") for i in attn]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            alignment = self.extract_alignment(
                attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos
            )
            finalized[i // beam_size][i % beam_size]["alignment"] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        src_tokens = (
            src_tokens[:, None, :]
                .expand(-1, self.beam_size, -1)
                .contiguous()
                .view(bsz * self.beam_size, -1)
        )
        src_lengths = sample["net_input"]["src_lengths"]
        src_lengths = (
            src_lengths[:, None]
                .expand(-1, self.beam_size)
                .contiguous()
                .view(bsz * self.beam_size)
        )
        prev_output_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]["attn"][0]
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn
