import os
import sys
from pathlib import Path

source=Path(sys.argv[1]).read_text()
def change(old,new):
    global source
    assert source.count(old)==1, old
    source=source.replace(old,new)

change("        image_tokens = tokens.shape[1]", "        if getattr(self, 'profile_compact', False):\n            return tokens, valid, attention\n        image_tokens = tokens.shape[1]")
change("    for step in range(plan['total_steps']):", """    events = []
    def event_pair():
        return [torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)]
    def wrap(name):
        fn = getattr(dist, name)
        def measured(output, input, *a, **kw):
            timer = event_pair()
            timer[0].record()
            result = fn(output, input, *a, **kw)
            timer[1].record()
            events.append((name, input.numel() * input.element_size(), timer))
            return result
        setattr(dist, name, measured)
    wrap('all_gather_into_tensor')
    wrap('reduce_scatter_tensor')
    for step in range(7):""")
change("        logged = torch.zeros(2, device=device)", """        logged = torch.zeros(2, device=device)
        model.module.profile_compact = step >= 4
        events.clear()
        input_wait = 0.0
        forward_timers, backward_timers = [], []""")
change("            rgb, ids, prompt_mask, state, actions, reliability = (x.to(device) for x in next(windows))", """            wait_start = time.monotonic()
            row = next(windows)
            input_wait += time.monotonic() - wait_start
            rgb, ids, prompt_mask, state, actions, reliability = (x.to(device) for x in row)""")
change("                flow, geom = model(obs, actions, reliability, noise, timestep, norms, args.condition)\n                ((flow + 100 * geom) / accum).backward()", """                fwd, bwd = event_pair(), event_pair()
                fwd[0].record()
                flow, geom = model(obs, actions, reliability, noise, timestep, norms, args.condition)
                fwd[1].record()
                bwd[0].record()
                ((flow + 100 * geom) / accum).backward()
                bwd[1].record()
                forward_timers.append(fwd)
                backward_timers.append(bwd)""")
change("        grad_norm = model.clip_grad_norm_(1.0)", """        post_timer = event_pair()
        post_timer[0].record()
        grad_norm = model.clip_grad_norm_(1.0)""")
change("        torch.musa.synchronize()", "        post_timer[1].record()\n        torch.musa.synchronize()")
change("            line = json.dumps(metric)", """            def secs(timers):
                return sum(a.elapsed_time(b) for a,b in timers) / 1000
            metric.update(compact=model.module.profile_compact, input_wait_seconds=input_wait,
                forward_seconds=secs(forward_timers), backward_seconds=secs(backward_timers),
                optimizer_and_post_seconds=secs([post_timer]))
            for name in ('all_gather_into_tensor', 'reduce_scatter_tensor'):
                matching = [(n,t) for op,n,t in events if op == name]
                metric[name] = dict(count=len(matching), seconds=secs(t for _,t in matching),
                    input_bytes=sum(n for n,_ in matching))
            line = json.dumps(metric)""")
path=Path(sys.argv[2])
path.write_text(source)
