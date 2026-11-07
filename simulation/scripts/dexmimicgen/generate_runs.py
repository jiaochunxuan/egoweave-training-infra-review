#!/usr/bin/env python
"""Print the six formal commands; never executes them."""
import argparse
from pathlib import Path
import sys
RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "src"))
from dexmg.specs import make_spec


def generate_commands(steps):
    rows=[]
    for group in ('A','B'):
        for variant in ('base','ego_our','egodexstyle'):
            make_spec(group,variant,steps)
            rows.append(f'bash {RELEASE_ROOT}/scripts/dexmimicgen/launch.sh --group {group} --variant {variant} --steps {steps} --gpus 0,1,2,3')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--dry-run',action='store_true',help='Always dry-run; accepted for clarity')
    args=parser.parse_args()
    print('# GPU IDs below are placeholders. Each command refuses occupied GPUs.')
    print('\n'.join(generate_commands(args.steps)))
