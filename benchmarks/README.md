# Moore throughput checks

The historical H100 JAX job used global batch 256 with 32 windows per device
on eight devices and one gradient computation per update. The first Moore
job split each device's 32 windows into eight microbatches of four.

| Eight-device configuration | Measured seconds/update | Basis |
| --- | ---: | --- |
| Microbatch 4, eight accumulation passes | 14.82 | Three post-startup updates with successive real batches |
| Microbatch 32, one pass | 8.69 | Two post-startup updates with successive real batches |
| Microbatch 32, controlled baseline | 8.76 | Median of two timed repetitions |
| Remove the masked wrist slots and trim trailing prompt padding | 6.04 | Same input, seed, weights, and objective |
| Also remove the outer transformer checkpoint | 5.41 | Per-layer checkpointing is active |
| Also reduce-scatter gradients in BF16 | 3.69 | FP32 optimizer state and geometric objective |

The four controlled configurations replay a fixed real batch of 256 windows
with the same random seed and learning rate set to zero. The first repetition
of each mode is excluded. The measured prompt tensor was trimmed from 768
to 640 on rank zero. The baseline and compact modes produced a total loss of
2.96250 and 2.96243 and a gradient norm of 2.55071 and 2.55821, respectively.
These checks characterize short-run numerical behavior.

With microbatch 4, the profiler observed eight parameter all-gathers taking
about 5.57 seconds per update. Microbatch 32 reduced this to one all-gather
taking about 0.69 seconds. The final optimization reduced the measured
reduce-scatter input from 14.47 GB to 7.23 GB and its timing from about
2.00 seconds to 0.26 seconds. Collectives are already included in the
end-to-end update timing.

The isolated attention probe used BF16 tensors, grouped-query attention, a
prefix/action mask, and head dimension 256. MUSA selected its math SDPA path
and reported that FlashAttention does not support this head dimension on
the tested build. The synthetic forward/backward median was 1.84 ms for the
eager operation and 1.15 ms for MUSA math SDPA. These are operator timings.

## Running the checks

Use the pinned Moore runtime described in the [repository README](../README.md).
Set the environment variables checked by [the launcher](run_moore.sh).
Run from the repository root to generate the instrumented entry points in
the same import scope as the training code.

```bash
python benchmarks/instrument_training.py \
    rebuttal/train_moore_torch.py rebuttal/_profile_training.py
python benchmarks/instrument_speed_ab.py \
    rebuttal/_profile_training.py rebuttal/_profile_speed_ab.py
EGOWEAVE_ENTRY="$PWD/rebuttal/_profile_speed_ab.py" \
    EGOWEAVE_MICROBATCH=32 bash benchmarks/run_moore.sh no_explicit_head
python benchmarks/sdpa_probe.py
```

The speed comparison mutates the FSDP handle's reduction dtype between modes
inside the diagnostic process. Set the desired dtype on `MixedPrecision`
when testing a fixed configuration. Timing logs are under [observed/](observed/).
