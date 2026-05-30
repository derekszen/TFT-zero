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
  --comp-detail-limit 12 \
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
  --epochs 500 \
  --out checkpoints/fight_value/current_patch_board_value.pt
```

`build_value_training_batch()` stacks encoded board states into NumPy arrays and
uses negative average placement as the target. The model is intentionally small:
it embeds units, items, active traits, augments, and target comp IDs, then trains
a scalar board-value head. This is the first bridge from MetaTFT aggregate data
to a planner/value model; it is not an RL policy yet.
