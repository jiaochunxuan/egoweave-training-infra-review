import sys
from pathlib import Path
source=Path(sys.argv[1]).read_text()
def change(old,new):
    global source
    assert source.count(old)==1, old
    source=source.replace(old,new)
change("    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):", """    def _apply_checkpoint(self, func, *args, **kwargs):
        if getattr(self, 'profile_drop_outer_checkpoint', False) and func.__name__ == 'forward_func':
            return func(*args, **kwargs)
        return super()._apply_checkpoint(func, *args, **kwargs)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):""")
change("    for step in range(7):", """    fixed_batch = next(windows)
    for step in range(12):
        mode = step // 3
        if step % 3 == 0:
            torch.musa.empty_cache()
        torch.manual_seed(seed + rank)
        torch.musa.manual_seed(seed + rank)
        model.module.profile_compact = mode >= 1
        model.module.profile_drop_outer_checkpoint = mode >= 2
        model._handle._reduce_dtype = torch.bfloat16 if mode >= 3 else torch.float32""")
change("        model.module.profile_compact = step >= 4\n", "")
change("            row = next(windows)", """            row = fixed_batch
            if mode >= 1:
                width = min(768, ((int(row[2].sum(-1).max()) + 63) // 64) * 64)
                row = (row[0], row[1][:, :width], row[2][:, :width], *row[3:])""")
change("        optimizer.param_groups[0]['lr'] = 2.5e-5 * min((step + 1) / 21, 1.0)", "        optimizer.param_groups[0]['lr'] = 0.0")
change("            metric.update(compact=model.module.profile_compact, input_wait_seconds=input_wait,", """            metric.update(mode=mode, prompt_width=ids.shape[1], compact=model.module.profile_compact,
                input_wait_seconds=input_wait,""")
Path(sys.argv[2]).write_text(source)
