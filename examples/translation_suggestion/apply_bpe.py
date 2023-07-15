import argparse
import sys
import fastBPE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes', type=str, default=None)
    parser.add_argument('--separator', type=str, default='@@')
    parser.add_argument('--debpe', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    bpe = fastBPE.fastBPE(args.codes)
    if not args.debpe:
        for line in sys.stdin:
            print(bpe.apply([line.strip()])[0])
    else:
        bpe_symbol = args.separator + " "
        for line in sys.stdin:
            print((line.strip() + " ").replace(bpe_symbol, '').strip())


if __name__ == '__main__':
    main()
