"""
Fix fine-tune JSON lines files, to be in the right format for the fine-tuning script.
"""

import sys

import orjson

if __name__ == '__main__':
    fname1 = sys.argv[1]
    fname2 = sys.argv[2]

    with open(fname1) as fd:
        fixed_lines = []
        for line in fd:
            data = orjson.loads(line)
            if isinstance(data, list) and len(data):
                fixed_lines.append({'messages': data})

    with open(fname2, 'wb') as fd:
        for line in fixed_lines:
            fd.write(orjson.dumps(line) + b'\n')
