import sys


def extract(prefix: str, suffix: str, mt: str):
    if prefix != "":
        if not mt.startswith(prefix):
            # print(f"MT should start with Prefix!\nMT: `{mt}`\nPrefix: `{prefix}`", file=sys.stderr, flush=True)
            return ""
        mt = mt[len(prefix):]
    if suffix != "":
        if not mt.endswith(suffix):
            # print(f"MT should end with Suffix!\nMT: `{mt}`\nSuffix: `{suffix}`", file=sys.stderr, flush=True)
            return ""
        mt = mt[:-len(suffix)]
    return mt.strip()


def main():
    prefix, suffix = None, None
    idx = -1
    for line in sys.stdin:
        info, content = line.split('\t', 1)
        new_idx = int(info.split('-')[1])
        if new_idx != idx:
            prefix = ""
            suffix = ""
        if info.startswith('C-'):
            if new_idx != idx:
                prefix = content.strip()
            else:
                suffix = content.strip()
        elif info.startswith('H-'):
            print(extract(prefix, suffix, content.split('\t', 1)[1].strip()))
        else:
            continue
        idx = new_idx


if __name__ == '__main__':
    main()
