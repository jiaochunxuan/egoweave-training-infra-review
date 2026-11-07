# EgoWeave training infrastructure review

This repository collects the training paths relevant to the EgoWeave human
pretraining and simulation experiments. The review focus is throughput and
correctness of the eight-device PyTorch port on Moore Threads hardware.

| Path | Contents |
| --- | --- |
| `rebuttal/` | PyTorch and JAX entry points, cached data reader, motion targets, geometric objective, and ablation checks |
| `benchmarks/` | Eight-device launch template, timing instrumentation, and measured results |
| `reference/` | Historical H100 training entry and OpenPI source excerpts used by the Moore trainer |
| `simulation/` | Existing pretraining release and downstream DexJoCo, DexMimicGen A/B, and RoboTwin adapters |

## Review starting points

- [PyTorch training loop](rebuttal/train_moore_torch.py) loads the matched base
  weights, sets up eight-device FSDP, and reports timing and peak device memory.
- [Structured objective](rebuttal/torch_objective.py) implements masked flow
  matching and geometry with the [shared target decoder](rebuttal/targets.py).
- [Cached reader](rebuttal/human_data.py) reads HDF5 JPEGs and NPZ pose packets.
  The historical 100-hour selection contains 4,710,394 windows and one epoch
  has 18,400 optimizer updates at global batch 256.
- [Performance notes](benchmarks/README.md) give the timings, controlled changes,
  and limits of the measurements.

## Runtime

The Moore environment used Python 3.10, PyTorch 2.7.1, torch_musa 2.7.1,
MUSA 4.3.5, MCCL 2.11.4, and the OpenPI-patched Transformers 4.53.2. Run
`benchmarks/run_moore.sh` from an activated compatible environment. The script
expects the project's OpenPI snapshot, an accessible local copy of the cached
data, condition-specific normalization, converted base weights, and the
seed-42 action interface. Paths in each dataset manifest must resolve on the
execution host.

The source excerpts in `reference/` are for inspection. The full OpenPI
runtime and experiment assets are separate inputs. Some historical JAX and
simulator entry points also expect companion experiment directories as
described by their launchers.
