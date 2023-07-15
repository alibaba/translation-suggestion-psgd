set -e
wd=`realpath $(dirname $0)`
FAIRSEQ_ROOT=`dirname $(dirname $wd)`

# Define input & output files
src_lang="de"
tgt_lang="en"
# original data files
src="${wd}/WeTS/corpus/de2en/de2en.test1.src"
mt_mask="${wd}/WeTS/corpus/de2en/de2en.test1.mask"
ts_ref="${wd}/WeTS/corpus/de2en/de2en.test1.tgt1"
# pre-processed input file
concat_input="${wd}/de2en.test1.input"
# generated output (whole sequence)
generate_output="${wd}/de2en.test1.output"
# extracted and de-tokenized Translation Suggestion result
ts_result="${wd}/de2en.test1.output.ts"

model_dir=${wd}/nmt_models/wmt19.de-en.joined-dict.ensemble
bpecodes=${model_dir}/bpecodes
model=${model_dir}/model1.pt

# Download WeTS corpus
git clone https://github.com/ZhenYangIACAS/WeTS.git

# Download WMT19 NMT de-en model
mkdir -p ${wd}/nmt_models
cd ${wd}/nmt_models

wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz
tar -zxf wmt19.de-en.joined-dict.ensemble.tar.gz

cd ${wd}
# tokenize & bpe src
cat ${src} \
  | python3 ${FAIRSEQ_ROOT}/examples/constrained_decoding/normalize.py --lang ${src_lang} \
  | python3 ${FAIRSEQ_ROOT}/examples/constrained_decoding/tok.py --lang ${src_lang} \
  | python3 $wd/apply_bpe.py --codes $bpecodes \
  > ${src}.bpe

cat ${mt_mask} \
  | sed $'s/<MASK_REP>/\t/g' \
  | cut -d$'\t' -f 1 \
  | python3  ${FAIRSEQ_ROOT}/examples/constrained_decoding/normalize.py --lang ${src_lang} \
  | python3  ${FAIRSEQ_ROOT}/examples/constrained_decoding/tok.py --lang ${src_lang} \
  | python3 $wd/apply_bpe.py --codes $bpecodes \
  | awk '{print "</s> " $0}' \
  > ${mt_mask}.prefix.bpe

cat ${mt_mask} \
  | sed $'s/<MASK_REP>/\t/g' \
  | cut -d$'\t' -f 2 \
  | python3  ${FAIRSEQ_ROOT}/examples/constrained_decoding/normalize.py --lang ${src_lang} \
  | python3  ${FAIRSEQ_ROOT}/examples/constrained_decoding/tok.py --lang ${src_lang} \
  | python3 $wd/apply_bpe.py --codes $bpecodes \
  | awk '{print $0 " </s>"}' \
  > ${mt_mask}.suffix.bpe


 # concat to input
 paste -d$'\t' ${src}.bpe ${mt_mask}.prefix.bpe ${mt_mask}.suffix.bpe > ${concat_input}.bpe

# PSGD inference
cat ${concat_input}.bpe \
  | python3 ${FAIRSEQ_ROOT}/fairseq_cli/interactive.py ${model_dir} \
      --task translation \
      --source-lang ${src_lang} --target-lang ${tgt_lang} \
      --path ${model} \
      --buffer-size 2000 --batch-size 64 \
      --beam 5 --remove-bpe="@@ " \
      --constraints --patience 5 \
      > ${generate_output}.psgd 2> ${generate_output}.psgd.log

# VDBA inference
cat ${concat_input}.bpe \
  | sed 's: </s>::g;s:</s> ::g' \
  | python3 ${FAIRSEQ_ROOT}/fairseq_cli/interactive.py ${model_dir} \
      --task translation \
      --source-lang ${src_lang} --target-lang ${tgt_lang} \
      --path ${model} \
      --buffer-size 2000 --batch-size 64 \
      --beam 5 --remove-bpe="@@ " \
      --constraints \
      > ${generate_output}.vdba 2> ${generate_output}.vdba.log

# extract the translation suggestion
cat ${generate_output}.psgd \
| python3 ${wd}/extract_ts.py \
| python3 ${wd}/moses_tokenize.py --lang ${tgt_lang} --detok \
> ${ts_result}.psgd

echo "PSGD prediction: " ${ts_result}.psgd

cat ${generate_output}.vdba \
| python3 ${wd}/extract_ts.py \
| python3 ${wd}/moses_tokenize.py --lang ${tgt_lang} --detok \
> ${ts_result}.vdba
echo "VDBA prediction: " ${ts_result}.vdba

# eval sacrebleu
sed 's/<NULL_REP>//g' ${ts_ref} > ${ts_ref}.ref
echo "PSGD BLEU: " `sacrebleu ${ts_ref}.ref --input ${ts_result}.psgd --tokenize "13a" --score-only`
echo "VDBA BLEU: " `sacrebleu ${ts_ref}.ref --input ${ts_result}.vdba --tokenize "13a" --score-only`
echo "PSGD TIME: " `grep "Total time" ${generate_output}.psgd.log  |cut -d' ' -f 8-`
echo "VDBA TIME: " `grep "Total time" ${generate_output}.vdba.log  |cut -d' ' -f 8-`