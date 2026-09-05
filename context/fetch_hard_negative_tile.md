# fetch_hard_negative_tile.py

Given a lat/lon, fetches the z17 tile that covers it (`DETECT_ZOOM`, matching
`oil_refinery/app/server/tile_server.py`) and prints the `HARD_NEGATIVE_TILES` line ready to paste
into `obb.py` — so a candidate hard negative spotted by eye on a map doesn't need a one-off script
each time. `common.fetch_tile` already caches it at `tiles/images/{z}_{x}_{y}.{ext}`, the exact
512px raw-tile format `HARD_NEGATIVE_TILES` expects; this just adds the lat/lon -> tile-id lookup
and a viewable copy (under `.scratch/hard_negative_previews/`) to check before committing it to
the list.

Loads `.env` itself (same manual-parse pattern `run.bat` uses, since nothing in this repo uses
python-dotenv) so it works as a standalone script regardless of which shell/profile it's launched
from — the project's `.venv` still needs to be the interpreter running it, though
(`.venv\Scripts\python.exe` on Windows), since a bare system `python` won't have `numpy`/`PIL`/etc.
installed.

Usage:

```
python scripts/fetch_hard_negative_tile.py --lat 51.259825 --lon 4.322423 --label "storage tank farm"
```
