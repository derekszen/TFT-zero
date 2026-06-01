# Current-Patch MetaTFT State Path

This path is separate from the Set-1-like toy simulator. It uses current-patch
MetaTFT aggregate data as a symbolic board/value layer.

## Catalog

`mini_tft.metatft.catalog` normalizes a MetaTFT comp-strength snapshot into:

- stable current-patch unit IDs with namespace `metatft_current_patch`
- source tags from comp names, classified as unit, trait, augment, archetype, or
  other
- comp records sorted by average placement, preserving source cluster IDs
- optional item, augment, trait-breakpoint, unit-item-build, and stage-line data
  from the richer MetaTFT endpoints

Load the current fixture:

```python
from pathlib import Path

from mini_tft.metatft import load_catalog_from_comp_strength

catalog = load_catalog_from_comp_strength(
    Path("tests/fixtures/metatft_set17_comp_strength_2026-05-31.json")
)
```

Fetch a richer current-patch snapshot:

```bash
uv run python -m mini_tft.tools.fetch_metatft_comp_strength \
  --rich \
  --comp-detail-limit 999 \
  --out data/metatft/current_rich_catalog.json \
  --min-count 3000
```

The rich fetch uses JSON endpoints exposed by MetaTFT:

- `comp_builds` for unit item builds
- `comp_options` for level 6-10 board/trait options
- `comp_augments` and static `itemEffects` for augment definitions
- `unit_items_processed` and `tft-stat-api/items` for item catalogs/stats
- per-comp `comp_details` for early options, trait levels, builds, and stage
  lines
- CommunityDragon's current TFT JSON for unit cost lookup by `apiName`

No HTML scraping is needed for this path right now, so `scrapling` is not a
runtime dependency.

## Board State

`CurrentBoardState` represents the state shape we need before using MetaTFT
value models for RL:

- stage and stage round
- level, gold, and HP
- board units with stars, positions, and optional items
- bench units
- active trait keys
- augment keys
- optional target comp ID

The schema supports level 10 and up to 12 board tokens so current-patch summons
or extra generated units can be represented without forcing them into the old
Set-1 integer ID space.

## Encoders

`CurrentBoardEncoder` emits NumPy arrays for value/planning models:

- scalar context
- board unit IDs, stars, positions, and item IDs
- bench unit IDs
- active trait IDs
- augment IDs
- target comp index

Final-board encoding:

```python
from mini_tft.metatft import CurrentBoardEncoder

encoder = CurrentBoardEncoder(catalog)
encoded = encoder.encode_final_board("409003")
```

Stage-line encoding:

```python
stage_lines = encoder.encode_stage_lines("409003")
```

When rich comp detail/options data exists, stage lines come from MetaTFT's real
early option and level-option rows. Without rich data, the fallback stage lines
are deterministic projections from the final comp:

- early: stage 2-1, first 4 units
- mid: stage 3-2, first 6 units
- late: stage 4-5, first 8 units
- final: stage 5-5, up to 10 units

Fallback projections are interface scaffolding, not real MetaTFT transition
traces.

## Value Training

The encoder is connected to a current-patch board-value training path:

```bash
uv run python -m mini_tft.rl.train_current_patch_value_model \
  --catalog data/metatft/current_rich_catalog.json \
  --device cuda \
  --blind-target-metadata \
  --epochs 100 \
  --embed-dim 16 \
  --hidden-dim 32 \
  --out checkpoints/fight_value/current_patch_board_value.pt
```

`build_value_training_batch()` stacks encoded board states into NumPy arrays and
uses negative average placement as the target. The model is intentionally small:
it embeds units, items, active traits, augments, and target comp IDs, then trains
a scalar board-value head. This is the first bridge from MetaTFT aggregate data
to a planner/value model; it is not an RL policy yet.

The training CLI now holds out 20% of comp IDs by default and reports final-board
ranking metrics on those heldout MetaTFT comps. Use `--blind-target-metadata`
when measuring this path; otherwise target comp rank/avg/top4/win scalars and
target comp IDs can leak the answer into the model input.

On the 2026-05-31 Set 17 snapshot, the best quick variant tried so far was a
small stage-line model with 100 epochs. It reached heldout pairwise accuracy
`0.643`, Spearman `0.407`, and top-4 overlap `0.50` across 14 heldout comps.

## Planner Scoring

`CurrentPatchPlannerScorer` loads a trained current-patch value checkpoint and
ranks candidate shop/bench/board transitions by the encoded value of the
resulting board state:

```python
from mini_tft.metatft import (
    CurrentPatchPlannerScorer,
    build_shop_bench_board_transitions,
)

scorer = CurrentPatchPlannerScorer.from_checkpoint(
    catalog,
    Path("checkpoints/fight_value/current_patch_board_value.pt"),
    device_name="cuda",
)
candidates = build_shop_bench_board_transitions(
    state,
    shop_unit_keys=("TFT17_Belveth", "TFT17_Ornn"),
)
ranked = scorer.rank_transitions(candidates)
```

This scorer is intentionally one-step: it evaluates candidate successor states.

## Shop/Econ Policy Shell

`CurrentPatchShopEconPolicy` wraps the scorer in a lightweight turn loop. It
generates candidate buy-to-bench, buy-to-board, field-bench, swap, buy-XP, roll,
target-board refill, and end-turn transitions, then asks the trained value
scorer to rank them. When `target_comp_id` and the current catalog are
available, it adds symbolic target-comp completion pressure so missing target
units, off-target extras, and duplicate filler do not beat a board that matches
the selected MetaTFT stage line.

```python
from mini_tft.metatft import CurrentPatchShopEconPolicy

policy = CurrentPatchShopEconPolicy.from_checkpoint(
    catalog,
    Path("checkpoints/fight_value/current_patch_board_value.pt"),
    device_name="cuda",
)
plan = policy.plan_turn(
    state,
    shops=[
        ("TFT17_Belveth", "TFT17_Ornn", "TFT17_MissFortune"),
        ("TFT17_Rhaast", "TFT17_Urgot", "TFT17_Kindred"),
    ],
    unit_costs={"TFT17_Ornn": 4, "TFT17_MissFortune": 4},
)
```

This is still a symbolic planner shell, not a complete TFT environment. Roll
actions consume the next provided shop in `shops`; the policy does not yet model
real current-patch shop odds, XP thresholds, augment offers, or item component
choices internally. The rich catalog includes current-patch unit costs when
CommunityDragon covers the set. If no unit-cost map is available for a unit,
shop buys use a conservative fallback cost of 3 gold so the econ loop does not
treat unknown units as free. The default policy ignores board/shop actions with
less than `0.02` predicted value improvement to avoid buying or swapping on
noise-sized score changes.

Run a checkpoint-backed policy smoke turn:

```bash
uv run python -m mini_tft.tools.plan_current_patch_turn \
  --catalog data/metatft/current_rich_catalog.json \
  --checkpoint checkpoints/fight_value/current_patch_board_value.pt \
  --device cuda
```

The smoke output includes `top_comp_match` metrics for level 8 and level 9 by
default. The metric compares the planner trace's final board against the top
MetaTFT comps with multiset unit overlap:

- `eligible`: whether the trace reached the target level
- `exact_match`: whether the final board exactly matches the target top-comp
  board at that level
- `partial_match`: whether recall is at or above `--min-recall`, default `0.75`
- `good_enough`: whether `exact_match` is true or recall is at least
  `--min-recall`
- `precision`, `recall`, and `jaccard`: overlap quality against the best top-k
  comp match

Use a higher demo level when smoke-testing the level 8/9 metric:

```bash
uv run python -m mini_tft.tools.plan_current_patch_turn \
  --catalog data/metatft/current_rich_catalog.json \
  --checkpoint checkpoints/fight_value/current_patch_board_value.pt \
  --device cuda \
  --demo-level 8 \
  --match-levels 8,9 \
  --top-k 10
```

For many fixed comp IDs, use the batch evaluator:

```bash
uv run python -m mini_tft.tools.evaluate_current_patch_planner \
  --catalog data/metatft/current_rich_catalog.json \
  --checkpoint checkpoints/fight_value/current_patch_board_value.pt \
  --device cuda \
  --comp-limit 16 \
  --demo-levels 8,9 \
  --match-levels 8,9 \
  --top-k 10 \
  --min-recall 0.75 \
  --out runs/current_patch_planner_eval.json
```

Use the same command as a regression gate before changing RL rewards/search.
The current smoke gate requires exact target-comp board completion at level 8
and level 9:

```bash
uv run python -m mini_tft.tools.evaluate_current_patch_planner \
  --catalog data/metatft/current_rich_catalog.json \
  --checkpoint checkpoints/fight_value/current_patch_board_value.pt \
  --device cuda \
  --comp-limit 8 \
  --demo-levels 8,9 \
  --match-levels 8,9 \
  --top-k 10 \
  --min-recall 0.75 \
  --require-exact-match-rate 8:1.0 \
  --require-exact-match-rate 9:1.0 \
  --out runs/current_patch_planner_gate.json
```

The command exits non-zero when a required metric drops below threshold, and
the JSON report includes `gate.failures` plus `exact_failure_summaries` for
debugging exact-match regressions.

The batch report summarizes per-level `exact_match_rate`,
`partial_match_rate`, `good_enough_rate`, and `eligible_good_enough_rate`.
The summary rows compare each demo level to its matching target level, while
the per-trace payload still includes every requested `match_level`.
Current default threshold:

```text
good_enough = eligible and (exact_match or recall >= 0.75)
```

Current exact-match investigation:

```text
Previous 8-comp smoke: exact_match_count = 0 at both level 8 and level 9.
Root cause: evaluator seeds used raw comp unit order while the metric used
MetaTFT stage-line targets; the one-step value planner could then prefer
duplicates or high-value substitutes.

Current 8-comp smoke after target seeding + target-refill planning:
level 8 exact_match_rate = 1.0
level 9 exact_match_rate = 1.0
```
