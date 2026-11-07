"""Build the 12-update fast-path check from the frozen Moore training entry."""
import sys
from pathlib import Path


source = Path(sys.argv[1]).read_text()


def change(old, new):
    global source
    if source.count(old) != 1:
        raise ValueError(f'Unexpected training source: {old}')
    source = source.replace(old, new)


for option in ('compact-padding', 'skip-outer-checkpoint'):
    change(f"parser.add_argument('--{option}', action='store_true')",
           f"parser.add_argument('--{option}', action='store_true', default=True)")
change('reduce_dtype=torch.float32', 'reduce_dtype=torch.bfloat16')
change("'gradient_reduce_dtype': 'float32'", "'gradient_reduce_dtype': 'bfloat16'")
change("    for step in range(plan['total_steps']):", "    for step in range(12):")
change("                          loss=flow + 100 * geom", "                          prompt_width=ids.shape[1], loss=flow + 100 * geom")
Path(sys.argv[2]).write_text(source)
