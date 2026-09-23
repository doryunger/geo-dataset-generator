# deploy

## nginx.conf

`location ~ \.mjs$` sets `default_type application/javascript` because nginx's stock `mime.types`
has no `.mjs` entry and would otherwise serve those files as `application/octet-stream`. Browsers
refuse to run an ES module worker with a non-JavaScript MIME type even on a 200, which is exactly
how maplibre-gl's worker (`/maplibre/*.mjs`) silently failed and left every GeoJSON layer on the
map blank -- full postmortem in `app/web/context/frontend.md`, under `maplibregl.setWorkerUrl`.
It's a regex location so it wins over the plain `/` and `/assets/` prefix blocks for any `.mjs`.
