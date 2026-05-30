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

## Simulator Hook

`FightValueNet` can now be selected as the simulator combat value path:

```python
from mini_tft.core.config import EnvConfig
from mini_tft.core.env import MiniTFTEnv

env = MiniTFTEnv(
    EnvConfig(
        combat_model="fight_value",
        fight_value_checkpoint="checkpoints/fight_value/set4_teacher_v1.pt",
        fight_value_device="cuda",
    )
)
```

The default remains `combat_model="abstract"`. Do not use the learned path for
RL rewards or search unless it passes the real-data ranking gate below.

## Real-Data Calibration

The current calibration fixture is a small MetaTFT current-patch snapshot from
May 31, 2026:

```text
tests/fixtures/metatft_set17_comp_strength_2026-05-31.json
```

Run a checkpoint against it:

```bash
uv run python -m mini_tft.tools.calibrate_fight_value_model \
  --checkpoint checkpoints/fight_value/set4_teacher_v1.pt \
  --fixture tests/fixtures/metatft_set17_comp_strength_2026-05-31.json \
  --device cuda
```

Use `--fail-on-threshold` in CI or promotion scripts. Passing means the model
meets the configured pairwise ranking, Spearman, and top-k overlap thresholds
against real comp placement data. Failing means the checkpoint is usable only as
a simulator/adapter smoke test, not as an RL value source.

## Current Boundary

The available checkpoint is trained on a vendored Set 4 teacher. The calibration
fixture is current Set 17 MetaTFT aggregate data. That mismatch is intentional:
it prevents us from accidentally treating teacher accuracy as real-meta
calibration.

## Current-Patch MetaTFT Adapter

The Set 4 teacher checkpoint should not be promoted as a current-patch value
model. For the current patch, fetch MetaTFT aggregate comp rankings and train a
separate ranking checkpoint with the current unit names embedded directly:

```bash
uv run python -m mini_tft.tools.fetch_metatft_comp_strength \
  --out data/metatft/current_comp_strength.json \
  --min-count 3000

uv run python -m mini_tft.rl.train_metatft_fight_value_model \
  --train-fixture data/metatft/current_comp_strength.json \
  --eval-fixture tests/fixtures/metatft_set17_comp_strength_2026-05-31.json \
  --device cuda \
  --epochs 4000 \
  --out checkpoints/fight_value/metatft_current_patch.pt
```

By default, the trainer excludes eval cluster IDs from the train split. Use the
same calibration command with `--fail-on-threshold` before wiring this checkpoint
into `EnvConfig(combat_model="fight_value")`. The
`--include-eval-clusters` flag is only for diagnostic overfit checks; it can
confirm the adapter can memorize current-patch comp rankings, but it is not a
holdout validation.

Current-patch MetaTFT checkpoints are tagged with the `metatft_current_patch`
unit namespace. They are valid for MetaTFT comp ranking/calibration, but they
are rejected by `MiniTFTEnv` until the simulator state is also current-patch and
uses the same unit-name mapping. This prevents Set 1-style integer unit IDs from
being scored as unrelated current-patch units.
