"""Eight-device MUSA pretraining with the matched EgoWeave objective."""
import argparse
from contextlib import nullcontext
import datetime
from itertools import islice
from importlib.metadata import version
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch_musa
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.utils.data import DataLoader, default_collate
from safetensors import safe_open
from safetensors.torch import load_file

# The vendor wheel uses Python 3.10. OpenPI uses the equivalent 3.11 alias.
if not hasattr(datetime, 'UTC'):
    datetime.UTC = datetime.timezone.utc

from human_data import MaskAfterNormalize, PacketDataset
from m1.protocol import EpochSampler, contiguous_rank_samples, epoch_plan
from openpi import transforms
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
from openpi.shared import normalize
from protocol import sha
from strict_tokenizer import StrictPaligemmaTokenizer
from torch_objective import losses


class TrainingWindows:
    def __init__(self, dataset, stats):
        self.dataset = dataset
        self.transform = transforms.compose([
            transforms.Normalize(stats),
            MaskAfterNormalize(),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(StrictPaligemmaTokenizer(768), discrete_state_input=True),
        ])

    def __getitem__(self, index):
        item = self.transform(self.dataset[index])
        return (item['image']['base_0_rgb'], item['tokenized_prompt'],
                item['tokenized_prompt_mask'], item['state'], item['actions'], item['reliability'])

    def __len__(self):
        return len(self.dataset)


def collate_without_prompt_padding(samples):
    rgb, ids, mask, state, actions, reliability = default_collate(samples)
    width = min(ids.shape[1], ((int(mask.sum(-1).max()) + 63) // 64) * 64)
    return rgb, ids[:, :width], mask[:, :width], state, actions, reliability


class StructuredPI05(PI0Pytorch):
    def _preprocess_observation(self, observation, *, train=True):
        obs = preprocess_observation_pytorch(observation, train=train, image_keys=('base_0_rgb',))
        return list(obs.images.values()), list(obs.image_masks.values()), obs.tokenized_prompt, obs.tokenized_prompt_mask, obs.state

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.config.skip_outer_checkpoint and func.__name__ == 'forward_func':
            return func(*args, **kwargs)
        return super()._apply_checkpoint(func, *args, **kwargs)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        tokens, valid, attention = super().embed_prefix(images, img_masks, lang_tokens, lang_masks)
        if self.config.compact_padding:
            return tokens, valid, attention
        image_tokens = tokens.shape[1] - lang_tokens.shape[1]
        # Reproduce the two masked wrist slots in the JAX attention layout.
        def pad_wrist_slots(values):
            slots = values.new_zeros((values.shape[0], 2 * image_tokens, *values.shape[2:]))
            return torch.cat((values[:, :image_tokens], slots, values[:, image_tokens:]), dim=1)
        return tuple(map(pad_wrist_slots, (tokens, valid, attention)))

    def forward(self, observation, actions, reliability, noise, timestep, norms, condition):
        predicted = []
        hook = self.action_out_proj.register_forward_hook(lambda _module, _inputs, result: predicted.append(result))
        try:
            with torch.autocast('musa', dtype=torch.bfloat16):
                super().forward(observation, actions, noise=noise, time=timestep)
        finally:
            hook.remove()
        velocity, = predicted
        flow, geom = losses(velocity, observation.state, actions, noise, timestep[:, None, None],
                            reliability, *norms, condition)
        return flow.mean(), geom.mean()


def initialize(base, interface, *, compact_padding, skip_outer_checkpoint):
    cfg = SimpleNamespace(pi05=True, action_dim=153, action_horizon=32, dtype='float32',
                          paligemma_variant='gemma_2b', action_expert_variant='gemma_300m',
                          pytorch_compile_mode=None, compact_padding=compact_padding,
                          skip_outer_checkpoint=skip_outer_checkpoint)
    model = StructuredPI05(cfg)
    state = load_file(str(base))
    with np.load(interface) as init:
        state['action_in_proj.weight'] = torch.from_numpy(init['action_in'])
        state['action_out_proj.weight'] = torch.from_numpy(init['action_out'])
        state['action_out_proj.bias'] = torch.from_numpy(init['action_bias'])
    # Safetensors stores one copy of tied weights and records the other names.
    with safe_open(str(base), framework='pt') as checkpoint:
        aliases = checkpoint.metadata() or {}
    for name in model.state_dict().keys() - state.keys():
        if aliases.get(name) in state:
            state[name] = state[aliases[name]]
    model.load_state_dict(state, strict=True)
    model.gradient_checkpointing_enable()
    return model


def save_checkpoint(model, optimizer, out, step, rank):
    target = out / str(step)
    staging = out / f'.{step}.staging'
    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.save({'model': model_state, 'optimizer': optim_state}, checkpoint_id=staging)
    if rank == 0:
        (staging / 'step.json').write_text(json.dumps({'completed_updates': step + 1}) + '\n')
        staging.rename(target)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=['no_explicit_head', 'current_head', 'no_scale'], required=True)
    for option in ('manifest', 'norm-dir', 'base', 'interface', 'output'):
        parser.add_argument(f'--{option}', type=Path, required=True)
    parser.add_argument('--microbatch', type=int, default=32)
    parser.add_argument('--batch-layout', choices=['contiguous', 'interleaved'], default='contiguous')
    parser.add_argument('--compact-padding', action='store_true')
    parser.add_argument('--skip-outer-checkpoint', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    batch, world, seed = 256, 8, 42
    if args.microbatch < 1 or (batch // world) % args.microbatch or int(os.environ['WORLD_SIZE']) != world:
        raise ValueError('Expected eight ranks and a microbatch dividing 32')
    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('musa', rank)
    dataset = PacketDataset(args.manifest, args.condition)
    provenance = json.loads((args.norm_dir / 'provenance.json').read_text())
    expected = dict(condition=args.condition, manifest_sha256=sha(args.manifest), selection_sha256=None,
                    norm_sha256=sha(args.norm_dir / 'norm_stats.json'), windows=len(dataset))
    if dataset.manifest.get('smoke') or any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError('Formal data provenance mismatch')
    stats = normalize.load(args.norm_dir)
    plan = epoch_plan(len(dataset), batch, 5)
    epoch = EpochSampler(len(dataset), batch, seed)
    sampler = contiguous_rank_samples(epoch, rank, world)
    if args.batch_layout == 'interleaved':
        sampler = islice(iter(epoch), rank, None, world)
    loader = DataLoader(TrainingWindows(dataset, stats), sampler=sampler, batch_size=args.microbatch,
                        num_workers=args.workers, prefetch_factor=2 if args.workers else None,
                        collate_fn=collate_without_prompt_padding if args.compact_padding else None)
    windows = iter(loader)
    torch.musa.set_device(rank)
    torch.manual_seed(seed + rank)
    dist.init_process_group('mccl')
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'config.json').write_text(json.dumps({
            **expected, **plan, 'world_size': world, 'seed': seed, 'microbatch': args.microbatch,
            'batch_layout': args.batch_layout, 'gradient_reduce_dtype': 'float32',
            'compact_padding': args.compact_padding, 'skip_outer_checkpoint': args.skip_outer_checkpoint,
            'base': str(args.base), 'interface': str(args.interface),
            'torch': torch.__version__, 'torch_musa': version('torch_musa'),
        }, indent=2) + '\n')
    dist.barrier()
    model = initialize(args.base, args.interface, compact_padding=args.compact_padding,
                       skip_outer_checkpoint=args.skip_outer_checkpoint)
    model = FSDP(model, device_id=device, use_orig_params=True, mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False,
        cast_root_forward_inputs=False))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5, betas=(.9, .95),
                                  eps=1e-8, weight_decay=1e-10, foreach=False, fused=False)
    norms = tuple(torch.as_tensor(getattr(stats[key], field), device=device, dtype=torch.float32)
                  for key in ('state', 'actions') for field in ('mean', 'std'))
    accum = batch // (world * args.microbatch)
    if rank == 0:
        print('FORMAL_READY', plan, flush=True)
    for step in range(plan['total_steps']):
        start = time.monotonic()
        torch.musa.reset_peak_memory_stats()
        logged = torch.zeros(2, device=device)
        for micro in range(accum):
            rgb, ids, prompt_mask, state, actions, reliability = (x.to(device) for x in next(windows))
            image = rgb.permute(0, 3, 1, 2).float() / 127.5 - 1
            obs = SimpleNamespace(images={'base_0_rgb': image}, image_masks={
                'base_0_rgb': torch.ones(args.microbatch, dtype=torch.bool, device=device)},
                tokenized_prompt=ids, tokenized_prompt_mask=prompt_mask, state=state,
                token_ar_mask=None, token_loss_mask=None)
            # Beta(1.5, 1) has the closed-form inverse CDF u**(2/3).
            timestep = torch.rand(args.microbatch, device=device).pow(2 / 3) * .999 + .001
            noise = torch.randn_like(actions)
            with model.no_sync() if micro < accum - 1 else nullcontext():
                flow, geom = model(obs, actions, reliability, noise, timestep, norms, args.condition)
                ((flow + 100 * geom) / accum).backward()
            logged.add_(torch.stack((flow.detach(), geom.detach())) / accum)
        grad_norm = model.clip_grad_norm_(1.0)
        dist.all_reduce(logged)
        flow, geom = (logged / world).tolist()
        grad_norm = grad_norm.item()
        if not np.isfinite((flow, geom, grad_norm)).all():
            raise FloatingPointError((step, flow, geom, grad_norm))
        optimizer.param_groups[0]['lr'] = 2.5e-5 * min((step + 1) / 21, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.musa.synchronize()
        peaks = torch.tensor([time.monotonic() - start, torch.musa.max_memory_allocated() / 2**30,
                              torch.musa.max_memory_reserved() / 2**30], device=device)
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        if rank == 0:
            seconds, allocated, reserved = peaks.tolist()
            metric = dict(step=step, seconds=seconds, examples_per_second=batch / seconds,
                          loss=flow + 100 * geom, flow=flow, geometry=geom, grad_norm=grad_norm,
                          peak_allocated_gib=allocated, peak_reserved_gib=reserved)
            line = json.dumps(metric)
            with (args.output / 'metrics.jsonl').open('a') as stream:
                stream.write(line + '\n')
            print(line, flush=True)
        if step in plan['checkpoint_steps']:
            save_checkpoint(model, optimizer, args.output, step, rank)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
