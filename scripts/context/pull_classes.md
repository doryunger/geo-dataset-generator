# pull_classes.sh / pull_classes.bat

Thin wrappers around `python scripts/pull_classes.py`, for people who'd rather double-click/run a
script than remember the venv path and `.env`-sourcing incantation (same reasoning as `run.bat` /
`install.sh` at the repo root).

`.env` sourcing is guarded with an existence check (`if [ -f .env ]` / `if exist .env`) rather than
sourced unconditionally like `restart.sh` does — `restart.sh` is only ever run against an already-set-up
machine, but a pull is plausibly the very first thing run on a fresh one, before `.env` exists. Letting
it through and hitting `pull_classes.py`'s own `s3_configured()` check gives a clearer error ("S3 not
configured (no S3_BUCKET_NAME)") than a shell-level "No such file or directory" from `source`.
