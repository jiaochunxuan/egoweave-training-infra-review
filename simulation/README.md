# EgoWeave: Chapter 4 simulation reproduction kit

This directory contains the public M4 structured-pretraining adapter,
robot-native fine-tuning code, and evaluation interfaces used for the
Chapter 4 simulation experiments. It is an experiment kit, not a copy of
the internal training workspace.

The release does not redistribute simulator data, human-motion records,
checkpoint weights, rollout videos, logs, or private service configuration.
All input and output locations are supplied through environment variables or
command-line arguments.

## What is included

- The paper-facing 153-D/156-D M4 representation, validity masks,
  normalization, and local21 geometry loss.
- M4 pretraining preparation, checkpoint/action-interface transfer, and the
  final pretraining configuration.
- Native-action dataset readers and fine-tuning adapters for DexJoCo,
  DexMimicGen A/B, and RoboTwin 2.0.
- Evaluation observation/action bridges, task and seed protocol, smoke probes,
  and public-contract tests.
- The reported optimizer settings and zero-based checkpoint-label convention.

The only human-motion pretraining row included is the final M4 recipe.
M0--M3 pretraining implementations and their intermediate assets are out of
scope.

## Reproduction protocol

The machine-independent protocol is in
configs/experiment_protocol.yaml. Unless a suite-specific command below
overrides it, downstream runs use Pi0.5, full-parameter fine-tuning, horizon
50, AdamW with learning rate 5e-5, 100 warm-up steps, seed 42, and no
online logging.

The final inference labels are:

- DexJoCo single-arm: 9999 (10000 optimizer updates);
- DexMimicGen A/B: 4999 (5000 optimizer updates);
- RoboTwin 2.0: 4999 (5000 optimizer updates).

The public task protocol uses the four DexJoCo tasks
(hammer_nail, pick_bucket, pinch_tongs, water_plant), all three tasks
in each DexMimicGen embodiment, and the five-task RoboTwin suite.

### Paper-row mapping

| Paper row | Initial checkpoint slot | DexJoCo | DexMimicGen | RoboTwin |
| --- | --- | --- | --- | --- |
| M0 | M0.pretrained | custom variant with the base checkpoint | base variant | custom variant with the base checkpoint |
| M1 | M1.pretrained | custom checkpoint path | egodexstyle for the official 48-D recipe | custom checkpoint path |
| M2 | M2.pretrained | custom checkpoint path | ego_our for the official 153-D recipe | custom checkpoint path |
| M3 | M3.pretrained | custom checkpoint path | custom checkpoint path | custom checkpoint path |
| M4 smoothed | M4_smoothed.pretrained | custom checkpoint path | custom checkpoint path | custom checkpoint path |
| M4 smoothed without geometry loss | M4_smoothed_off.pretrained | custom checkpoint path | custom checkpoint path | custom checkpoint path |

Initialization is deliberately a command-line/configuration choice. This
keeps private run names and checkpoint locations out of the source while
preserving the comparison between rows. Use the same downstream recipe,
normalization, task subset, seed list, and final label within each suite.

## Installation and external boundary

Install a compatible public OpenPI/Pi0.5 checkout, JAX/Flax/Optax runtime,
and the official simulator/evaluator packages separately. Install this
package with:

~~~bash
python -m pip install -e '.[simulation]'
python -m pip install -e '.[pretrain]'
~~~

The simulation extra supplies Parquet/video/image readers and the RoboTwin
image adapter (av, pyarrow, Pillow, and PyTorch). The pretrain extra
supplies the HDF5 and image-cache readers used by M4 preparation.

Set local locations through environment variables or CLI flags. These are
placeholders only:

~~~bash
export EGO_WEAVE_ROOT=/path/to/this/release
export DEXJOCO_OPENPI_ROOT=/path/to/openpi
export DEXJOCO_SOURCE_ROOT=/path/to/dexjoco
export DEXJOCO_DATA_ROOT=/path/to/dexjoco_lerobot_datasets
export ROBOTWIN_OPENPI_ROOT=/path/to/openpi
export ROBOTWIN_DATA_ROOT=/path/to/robotwin_dataset
export LEROBOT_ROOT=/path/to/lerobot
export PI05_BASE_PARAMS=/path/to/pi05_base/params
~~~

The companion reconstruction checkout, OpenPI/Pi0.5, DexJoCo,
DexMimicGen/robomimic, RoboTwin 2.0, and LeRobot remain separate public
dependencies. Their licenses and source revisions must be reviewed in their
own distributions. This release does not redistribute those projects,
weights, or datasets.

## Final M4 pretraining

The final M4 recipe uses 153-D structured actions, 156-D state,
EgoDexPro records exported by the companion reconstruction pipeline,
mean/std normalization, availability masks for incomplete hands, and the
local21 geometry loss. The entry point is
src/egoweave_pretrain/train_m4.py, with defaults in
configs/m4_pretraining.yaml.

### Data contract

Export one .npz packet and one 224x224 RGB HDF5 cache per episode. The
packet contains:

~~~text
pose153           float32 [T,162]
anchors           int array [N]
camera_valid      bool [T]
head_valid        bool [T]
hand_valid        bool [T,2]
alignment_version scalar string: m4_153d_camera_head_progress_local21_v1
~~~

The HDF5 cache contains anchors, jpeg_offsets, jpeg_bytes, and a
header_json attribute with committed: true, an episode key, and frame
count. Create a JSONL index with key, packet_path, image_cache_path, and
prompt, then build the portable manifest and normalization file:

~~~bash
PYTHONPATH=src python -m egoweave_pretrain.prepare \
  --index /path/to/episode_index.jsonl \
  --output /path/to/m4_manifest \
  --data-root /path/to/data
~~~

The generated manifest and normalization files are inputs, not files to
commit to the code release. Every packet and cache is hash-checked before a
worker opens it. Invalid hand labels become zero placeholders with a
per-channel mask and are excluded from normalization statistics.

### Model-side overlay contract

The simulator adapters use an unmodified public OpenPI/Pi0.5 checkout. The
M4 pretraining row needs the following small model-side additions:

1. Carry Observation.reliability through Observation.from_dict and
   preprocess_observation until Pi0.compute_loss; it is the per-channel
   target-validity mask.
2. Add the following fields to Pi0Config:

~~~python
image_keys: tuple[str, ...] = ("base_0_rgb",)
preserve_image_token_layout: bool = False
action_loss_weights: tuple[float, ...] | None = None
flow_loss_type: str = "mse"
m4_local21_recon_loss_weight: float = 0.0
m4_local21_recon_left_weight: float = 1.0
m4_local21_recon_right_weight: float = 1.0
m4_local21_recon_left_scale: float = 1.0
m4_local21_recon_right_scale: float = 1.0
m4_local21_recon_wrist_pose_relative_to_head: bool = False
m4_local21_recon_wrist_position_absolute: bool = False
m4_local21_recon_wrist_rotation_absolute: bool = False
m4_local21_recon_state_mean: tuple[float, ...] | None = None
m4_local21_recon_state_std: tuple[float, ...] | None = None
m4_local21_recon_action_mean: tuple[float, ...] | None = None
m4_local21_recon_action_std: tuple[float, ...] | None = None
~~~

Validate the action-weight length against action_dim and require the four
statistics whenever the geometry weight is positive. The public M4 run uses
image_keys=("base_0_rgb",) and preserve_image_token_layout=True.

3. Keep upstream flow-matching noise sampling and model forward unchanged.
   Reduce the predicted velocity error with
   weighted_flow_matching_loss and the recipe's action weights, convert the
   velocity to the implied clean action, and add the local21 reconstruction
   loss from src/egoweave_pretrain/geometry_loss.py with geometry weight
   100.0.

train_m4.py checks these fields before starting distributed training and
fails with an actionable message if they are absent. The model-side hook is
an integration contract rather than a vendored OpenPI fork; review it against
the selected public OpenPI revision after upgrades.

### Preflight and run

Run the CPU-only contract check first:

~~~bash
PYTHONPATH=src python -m egoweave_pretrain.train_m4 \
  --manifest /path/to/m4_manifest/manifest.json \
  --norm-stats /path/to/m4_manifest/norm_stats.json \
  --preflight-only
~~~

The distributed entry point is scripts/pretrain_m4.sh. It uses one process
with eight visible devices, global batch 256 (32 per device), AdamW with
learning rate 2.5e-5, warm-up 20, seed 42, one shuffled epoch, and
zero-based checkpoint labels. The reported run uses 18400 updates and label
18399.

## Downstream fine-tuning

Run a CPU-only probe before allocating accelerators:

~~~bash
PYTHONPATH=src python -m dexjoco4.train.train_dexjoco_pi05 \
  --variant custom --custom-variant smoke \
  --checkpoint-path "$PI05_BASE_PARAMS" --dry-run

PYTHONPATH=src python scripts/dexmimicgen/run_train.py \
  --group A --variant base --steps 3 --smoke --gpus 0,1

PYTHONPATH=src python src/robotwin_pi05/train_robotwin_pi05.py \
  --variant base --steps 3 --probe
~~~

Formal launchers do not select machines or GPUs. Set
CUDA_VISIBLE_DEVICES outside the launcher and pass an explicit step budget.

### DexJoCo

~~~bash
export DEXJOCO_DATA_ROOT=/path/to/dexjoco_lerobot_datasets
export DEXJOCO_OPENPI_ROOT=/path/to/openpi
export EGO_PRETRAIN_PARAMS=/path/to/pretrained/params
export DEXJOCO_OUTPUT_ROOT=/path/to/output/dexjoco
export CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHONPATH=src python -m dexjoco4.train.train_dexjoco_pi05 \
  --variant custom --custom-variant M4_smoothed \
  --checkpoint-path "$EGO_PRETRAIN_PARAMS" \
  --data-root "$DEXJOCO_DATA_ROOT" \
  --output-root "$DEXJOCO_OUTPUT_ROOT" \
  --steps 10000 --save-interval 10000 --keep-period 10000 --overwrite
~~~

The final checkpoint directory is labelled 9999 because labels are
zero-based. Pass that label to the DexJoCo evaluator and use the four tasks
in the protocol.

### DexMimicGen A/B

Prepare one formal manifest and normalization asset for each group, then
launch each initialization independently:

~~~bash
export DEXMG_OPENPI_ROOT=/path/to/openpi_snapshot
export PI05_BASE_PARAMS=/path/to/pi05_base/params
export EGO_OUR_PARAMS=/path/to/ego_our/params
export EGO_STYLE_PARAMS=/path/to/ego_style/params
PYTHONPATH=src python scripts/dexmimicgen/prepare_data.py \
  --group A --data-root /path/to/dexmimicgen/raw \
  --output /path/to/manifest/A.json
PYTHONPATH=src python scripts/dexmimicgen/run_train.py \
  --group A --variant ego_our --steps 5000 --gpus 0,1,2,3
~~~

Use the same explicit budget for group B. The final label is 4999.
The A and B native action interfaces remain separate: A is 24-D and B is
30-D during training; the B rollout adapter converts rotation-6D back to the
simulator's native representation before env.step.

### RoboTwin 2.0

~~~bash
export ROBOTWIN_OPENPI_ROOT=/path/to/openpi
export ROBOTWIN_DATA_ROOT=/path/to/robotwin_dataset
export ROBOTWIN_OUTPUT_ROOT=/path/to/output/robotwin
export CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHONPATH=src python src/robotwin_pi05/train_robotwin_pi05.py \
  --variant custom --custom-variant M4_smoothed \
  --checkpoint-path /path/to/m4_smoothed/params \
  --data-root "$ROBOTWIN_DATA_ROOT" \
  --output-root "$ROBOTWIN_OUTPUT_ROOT" \
  --steps 5000 --save-interval 1000 --overwrite
~~~

The five task names and fixed-seed policy are in
configs/experiment_protocol.yaml. The official RoboTwin evaluator remains
external; this release contains the training data adapter and model interface
used to create its checkpoint.

## Evaluation

DexJoCo evaluation is launched with
src/dexjoco4/eval/eval_dexjoco4_pi05.py; it starts the OpenPI websocket
server and calls the installed DexJoCo client. DexMimicGen A/B and RoboTwin
use their official environment evaluators. This package provides the matching
observation/action adapters, task sets, seeds, and checkpoint interface, but
does not copy the simulator packages.

## Release boundary and privacy

The following paper-critical material is included: representation and masks,
geometry loss, data/normalization contracts, model-interface transfer,
fine-tuning schedules, evaluation adapters, and protocol configuration.

The following are deliberately absent: checkpoint payloads, private datasets,
internal human-motion assets, machine inventory, remote host aliases, user
names, credentials, experiment logs, aggregate success-rate tables, rollout
outputs, internal auxiliary methods, cluster launchers, monitoring,
deployment utilities, and absolute workstation/server paths. Generated
manifests, download receipts, normalization provenance, logs, checkpoints, and
rollouts should not be committed because they can record local paths.

## Local release check

Run the dependency-free release gate and the public-contract tests before
publishing:

~~~bash
python -B scripts/check_release.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q -p no:cacheprovider
~~~

The release gate checks required files, private-path/infrastructure patterns,
and generated Python artifacts. It intentionally does not inspect the
external simulator or OpenPI installations.
