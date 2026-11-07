"""Build the 12-update fast-path check from the frozen Moore training entry."""
import sys
from pathlib import Path


source = Path(sys.argv[1]).read_text()


def change(old, new):
    global source
    if source.count(old) != 1:
        raise ValueError(f'Unexpected training source: {old}')
    source = source.replace(old, new)


change("    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):", """    def _apply_checkpoint(self, func, *args, **kwargs):
        if func.__name__ == 'forward_func':
            return func(*args, **kwargs)
        return super()._apply_checkpoint(func, *args, **kwargs)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):""")
start = source.index('        image_tokens = tokens.shape[1]')
end = source.index('\n    def forward(', start)
source = source[:start] + '        return tokens, valid, attention\n' + source[end:]
change('reduce_dtype=torch.float32', 'reduce_dtype=torch.bfloat16')
change("    for step in range(plan['total_steps']):", "    for step in range(12):")
change("            rgb, ids, prompt_mask, state, actions, reliability = (x.to(device) for x in next(windows))", """            row = next(windows)
            width = min(768, ((int(row[2].sum(-1).max()) + 63) // 64) * 64)
            row = (row[0], row[1][:, :width], row[2][:, :width], *row[3:])
            rgb, ids, prompt_mask, state, actions, reliability = (x.to(device) for x in row)""")
change("                          loss=flow + 100 * geom", "                          prompt_width=ids.shape[1], loss=flow + 100 * geom")
Path(sys.argv[2]).write_text(source)
