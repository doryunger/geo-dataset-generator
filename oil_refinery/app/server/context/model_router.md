# model_router.py

Decides which models run against an incoming tile — never which classes within a model to look
for (see `oil_refinery/semantic_graph.md`'s "Pipeline: model router, fuser, classifier"). Every
triggered model runs unfiltered, returning whatever classes it detects; nothing here or downstream
restricts a model's own class list — that was `config.json`'s old per-target `class_id` filter,
dropped along with this module's earlier version.

Today's only routing criterion is the zoom gate `server.py` already used: below `MIN_DETECT_ZOOM`,
detection doesn't run at all, so no model is worth triggering; at or above it, every configured
model runs. Room for additional per-tile routing criteria later, but nothing beyond zoom exists yet.

`CANONICAL_MODEL` is the fuser's fixed naming/tie-break convention (see `semantic_graph.md`'s
"Pipeline") — read from `config.json` here since this module already owns that file, but only
`fuser.py` actually uses the value.
