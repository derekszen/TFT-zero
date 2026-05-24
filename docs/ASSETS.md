# Assets

Assets are not a v0 dependency.

## v0

Use text IDs and placeholder colors only.

The RL environment should not require thumbnails, icons, sprites, or a browser
renderer.

## v1

Add optional debug assets:

```text
assets/
  units/
    unit_id.png
  items/
    item_id.png
```

Map `unit_id` and `item_id` to paths in the renderer or dashboard layer. Do not
put image paths into the core simulator state.

## Acquisition

Possible sources:

- official static endpoints, when available
- community-maintained data mirrors
- extracted asset repositories
- fan wikis as a last resort
- generated placeholder icons

Cache assets locally and keep the cache out of git unless the asset license
explicitly allows redistribution.

## V0 Scraper

The local scraper downloads base champion square icons from Riot Data Dragon for
the current 24-unit roster:

```bash
uv run python -m mini_tft.tools.scrape_assets
```

The default Data Dragon version is `9.19.1`, matching the Set 1-era flavor. The
script writes `assets/units/*.png` and updates `assets/manifest.json`.

## Legal And Practical Notes

- Use third-party game art only for local research and debug UI unless usage
  terms permit broader distribution.
- Avoid packaging large asset dumps into releases.
- Generated or original placeholder icons are safest for public demos.
