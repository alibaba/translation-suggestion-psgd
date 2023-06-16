# PSGD
The code for ACL 2023 paper [Easy Guided Decoding in Providing Suggestions for Interactive Machine Translation](https://arxiv.org/abs/2211.07093).

The code is forked from [this commit of fairseq](https://github.com/facebookresearch/fairseq/tree/e0884db9).

PSGD is short for "Prefix-Suffix Guided Decoding". 
It's a constrained decoding algorithm that generate a span in a sequence with given prefix and suffix. 
We propose and apply this algorithm for Translation Suggestion, improving both suggestion quality and computation efficiency.


# Getting Started
An example script of running PSGD and compare it with the [Lexically Constrained Decoding](examples/constrained_decoding/README.md) on a Translation Suggestion dataset can be found in [examples/translation_suggestion/example.sh](examples/translation_suggestion/example.sh).


# Requirements
See the [README of fairseq](https://github.com/facebookresearch/fairseq/tree/e0884db9#requirements-and-installation) for details

# License
PSGD is MIT-licensed.

# Citation

Please cite as:

``` bibtex
@inproceedings{wang2023easy,
    title={Easy Guided Decoding in Providing Suggestions for Interactive Machine Translation},
    author={Ke Wang and Xin Ge and Jiayi Wang and Yu Zhao and Yuqi Zhang},
    year={2023},
    booktitle={Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)}
}
```
