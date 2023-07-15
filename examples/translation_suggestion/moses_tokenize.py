import argparse
import sys

from sacremoses import MosesTokenizer, MosesDetokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--escape', default=False, action='store_true')
    parser.add_argument('--dash_splits', default=False, action='store_true')
    parser.add_argument('--detok', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.detok:
        tokenizer = MosesTokenizer(args.lang).tokenize
        for line in sys.stdin:
            toked = tokenizer(
                line.strip(),
                aggressive_dash_splits=args.dash_splits,
                escape=args.escape,
                return_str=True
            )
            print(toked)
    else:
        detokenizer = MosesDetokenizer(args.lang).detokenize
        for line in sys.stdin:
            detoked = detokenizer(line.strip().split())
            print(detoked)


if __name__ == '__main__':
    main()
