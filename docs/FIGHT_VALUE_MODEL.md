# Fight Value Model

This pipeline uses detailed TFT combat as a CPU teacher and trains a GPU-friendly
`FightValueNet` that can later replace expensive combat calls in rollouts.

## Teacher

The default teacher is the vendored `third_party/TFTMuZeroAgent` Set 4 combat
simulator. It runs detailed Python combat through a narrow adapter and writes
fixed-schema fight labels. A `heuristic` teacher is available only for smoke
tests and pipeline debugging.

## Generate Labels

Small smoke:

```bash
uv run python -m mini_tft.tools.generate_fight_labels \
  --teacher heuristic \
  --target-fights 200 \
  --workers 2 \
  --shard-size 50 \
  --out /tmp/fight_labels_smoke
```

Teacher run:

```bash
uv run python -m mini_tft.tools.generate_fight_labels \
  --teacher tft-muzero \
  --target-fights 1000000 \
  --workers 12 \
  --shard-size 10000 \
  --out data/fight_labels/set4_teacher_v1
```

The dataset is stored as Zarr shards plus a `manifest.json`. Completed shards are
skipped on resume.

## Train

```bash
uv run python -m mini_tft.rl.train_fight_value_model \
  --dataset data/fight_labels/set4_teacher_v1 \
  --device cuda \
  --epochs 10 \
  --batch-size 16384 \
  --out checkpoints/fight_value/set4_teacher_v1.pt
```

## Benchmark

```bash
uv run python -m mini_tft.tools.benchmark_fight_value_model \
  --checkpoint checkpoints/fight_value/set4_teacher_v1.pt \
  --device cuda \
  --batch-size 16384 \
  --batches 100
```

## Current Boundary

This does not alter the main simulator combat path yet. The intended next step
is validating held-out teacher accuracy and GPU throughput before wiring
`FightValueNet` into RL rollouts.
