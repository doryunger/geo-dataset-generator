import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { ExtentSocket, type SiteFeatureCollection, type SiteFeatureProperties } from './api'

// The floor below which nothing here does anything at all, not even reporting a live view --
// deliberately below the server's DETECT_ZOOM (tile_server.py, currently 17), not
// equal to it: the live view still always resolves to DETECT_ZOOM tiles regardless of what zoom the
// user is actually at (see visibleDetectZoomTiles()), so starting the trigger a level earlier just
// means detection kicks in a little before the display zoom catches up to DETECT_ZOOM itself. Kept
// 2 levels below DETECT_ZOOM rather than 1 (as when DETECT_ZOOM was 16) would mean 16x the tiles per
// viewport report at this floor (2^(DETECT_ZOOM - MIN_DETECT_ZOOM) per axis) -- worth reconsidering
// if browsing right at this floor turns out to feel slow with the queue this generates.
const MIN_DETECT_ZOOM = 15

// Matches the server's DETECT_ZOOM (tile_server.py) -- the *only* zoom real detection ever runs at.
// The live-view report below is always computed directly at this zoom, regardless of what zoom the
// map is actually displaying, so the tiles sent are exactly what intersects the current screen at
// DETECT_ZOOM resolution -- not "whatever tile the current zoom's grid happens to cover," which
// could include a lot of ground that's barely touching the edge of the viewport, not actually on it.
const DETECT_ZOOM = 17

// How much of the viewport's edge margin to skip, as a fraction of width/height -- scales with how
// far below DETECT_ZOOM the map is currently displayed, not a fixed constant: full MAX_VIEWPORT_TRIM
// right at the MIN_DETECT_ZOOM floor (where the tile count is worst, see MIN_DETECT_ZOOM's comment),
// tapering linearly to 0 by the time the display zoom reaches DETECT_ZOOM itself (where the tile
// count is already sane and trimming would just lose coverage for no benefit). A prior version of
// this trim used one fixed fraction at every zoom; scaling it by zoom gap instead means it only
// costs coverage where the tile count actually needs it.
const MAX_VIEWPORT_TRIM = 0.3

function viewportTrimFraction(displayedZoom: number): number {
  const maxGap = DETECT_ZOOM - MIN_DETECT_ZOOM
  if (maxGap <= 0) return 0
  const gap = DETECT_ZOOM - displayedZoom
  return Math.min(MAX_VIEWPORT_TRIM, Math.max(0, MAX_VIEWPORT_TRIM * (gap / maxGap)))
}

// Same Hamburg refinery site already used elsewhere in this repo's own probing
// (oil_refinery/probe_pretrained.py) -- a known-good spot with real storage tanks to look at.
// Deliberately *below* MIN_DETECT_ZOOM -- loading the page shouldn't immediately fire off a big
// processing batch before the user has actually chosen to zoom in on anything.
const INITIAL_CENTER: [number, number] = [9.9517431, 53.4770211]
const INITIAL_ZOOM = 14

const EMPTY_FEATURE_COLLECTION: SiteFeatureCollection = { type: 'FeatureCollection', features: [] }

// Standard slippy-map tile math (Web Mercator) -- same tiling scheme the server's own
// common.lonlat_to_tile already uses, just computed here so the map's *current view* can be
// reported without round-tripping through the tile-loading events MapLibre already fires.
function lonLatToTile(lon: number, lat: number, z: number): [number, number] {
  const n = 2 ** z
  const x = Math.floor(((lon + 180) / 360) * n)
  const latRad = (lat * Math.PI) / 180
  const y = Math.floor(((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n)
  return [x, y]
}

// Every DETECT_ZOOM tile whose bounds actually intersect the current viewport -- computed straight
// from the screen bounds at DETECT_ZOOM, not by taking whatever tile the *displayed* zoom's grid
// covers and expanding it to every descendant (which would include plenty of ground nowhere near
// the screen whenever a displayed-zoom tile only partially overlaps the viewport's edge).
//
// Bounds are shrunk by `trimFraction` first (defaults to viewportTrimFraction(map.getZoom()) when
// not given explicitly) -- an even margin trimmed off each side, not the whole box scaled from a
// corner -- so the skipped strip stays centered around the edges the user is least likely to be
// looking directly at. Passing 0 explicitly gets the untrimmed full viewport -- see the two-stage
// send in the component below (trimmed first, then a full-coverage follow-up).
function visibleDetectZoomTiles(
  map: maplibregl.Map, trimFraction: number = viewportTrimFraction(map.getZoom()),
): { x: number; y: number }[] {
  const bounds = map.getBounds()
  const west = bounds.getWest()
  const east = bounds.getEast()
  const south = bounds.getSouth()
  const north = bounds.getNorth()
  const trim = trimFraction / 2
  const lonInset = (east - west) * trim
  const latInset = (north - south) * trim

  const [xMin, yMin] = lonLatToTile(west + lonInset, north - latInset, DETECT_ZOOM)
  const [xMax, yMax] = lonLatToTile(east - lonInset, south + latInset, DETECT_ZOOM)
  const tiles: { x: number; y: number }[] = []
  for (let x = xMin; x <= xMax; x++) {
    for (let y = yMin; y <= yMax; y++) {
      tiles.push({ x, y })
    }
  }
  return tiles
}

interface LabelFeatureCollection {
  type: 'FeatureCollection'
  features: {
    type: 'Feature'
    geometry: { type: 'Point'; coordinates: [number, number] }
    properties: SiteFeatureProperties
  }[]
}

// One Point feature per site (at its label point) derived from the polygon FeatureCollection the
// server sends -- a symbol layer needs its own point source, it can't place text from a Polygon
// source's vertices.
function labelsFrom(fc: SiteFeatureCollection): LabelFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: fc.features.map((f) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [f.properties.label_lon, f.properties.label_lat] },
      properties: f.properties,
    })),
  }
}

export default function Map() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [zoom, setZoom] = useState(INITIAL_ZOOM)
  const [sites, setSites] = useState<SiteFeatureCollection>(EMPTY_FEATURE_COLLECTION)

  useEffect(() => {
    if (!containerRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        // Public, tokenless glyph server -- needed for the "site-label" symbol layer's text below;
        // without a `glyphs` source MapLibre has no font data at all, so text-field silently never
        // renders anything (confirmed by direct testing: the layer was configured correctly and
        // the polygon drew fine, but no label text ever appeared until this was added).
        glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
        sources: {
          // Two independent sources stacked as two layers, not one -- base imagery is never held
          // up by how long detection takes (it's always fetched and rendered on its own), and the
          // transparent detections overlay just pops in on top whenever its tile finishes. Both
          // point at our own backend (not Mapbox directly): MapLibre's normal tile-loading
          // behavior (figure out visible tiles, fetch, cache, abort stale fetches on pan/zoom) is
          // the entire trigger mechanism for both. See tile_server.py's two endpoints.
          basemap: {
            type: 'raster',
            tiles: ['/api/tile/{z}/{x}/{y}'],
            tileSize: 512,
            // Capped at DETECT_ZOOM so MapLibre stops requesting new tiles past it and instead
            // scales up the last-fetched DETECT_ZOOM tile (standard TileJSON overzoom behavior) --
            // matches the 'detections' source's cap below so the two layers stay pixel-aligned at
            // every zoom, and specifically avoids ever fetching a native-resolution tile above
            // DETECT_ZOOM that the (capped) detections overlay can no longer line boxes up against.
            maxzoom: DETECT_ZOOM,
            attribution: '© Mapbox',
          },
          detections: {
            type: 'raster',
            tiles: ['/api/detections/{z}/{x}/{y}'],
            tileSize: 512,
            // Same cap as 'basemap' above -- without it, MapLibre requests /api/detections tiles
            // above DETECT_ZOOM, which the server always answers transparent (tile_server.py's
            // get_detections), so zooming in past DETECT_ZOOM used to make every detection box
            // silently disappear. Capping here means MapLibre instead upscales the last real
            // DETECT_ZOOM overlay it already has, so the boxes stay visible (just blockier) instead
            // of vanishing.
            maxzoom: DETECT_ZOOM,
          },
        },
        layers: [
          { id: 'basemap', type: 'raster', source: 'basemap' },
          { id: 'detections', type: 'raster', source: 'detections' },
        ],
      },
      center: INITIAL_CENTER,
      zoom: INITIAL_ZOOM,
    })
    map.addControl(new maplibregl.NavigationControl())
    map.on('zoom', () => setZoom(map.getZoom()))

    // Dev/debug convenience -- direct console access to the live MapLibre instance (e.g.
    // `map.getSource('site-boundaries')._data`, `map.getZoom()`) without threading it through
    // React state anywhere.
    ;(window as typeof window & { map?: maplibregl.Map }).map = map

    map.on('load', () => {
      // Identified-site boundaries + labels -- a third, independent layer on top of the two raster
      // ones above. Fed by the websocket below, not by tile loading (see api.ts's ExtentSocket).
      map.addSource('site-boundaries', { type: 'geojson', data: EMPTY_FEATURE_COLLECTION })
      map.addLayer({
        id: 'site-fill', type: 'fill', source: 'site-boundaries',
        paint: { 'fill-color': '#ffee00', 'fill-opacity': 0.55 },
      })
      map.addLayer({
        id: 'site-outline', type: 'line', source: 'site-boundaries',
        paint: { 'line-color': '#ff00aa', 'line-width': 4 },
      })
      map.addSource('site-labels', { type: 'geojson', data: labelsFrom(EMPTY_FEATURE_COLLECTION) })
      map.addLayer({
        id: 'site-label', type: 'symbol', source: 'site-labels',
        layout: { 'text-field': ['get', 'site'], 'text-size': 16, 'text-font': ['Open Sans Semibold'] },
        paint: { 'text-color': '#fff', 'text-halo-color': '#000', 'text-halo-width': 1.5 },
      })
    })

    // Reports the live view on every moveend (pan/zoom settled) -- see api.ts's ExtentSocket and
    // semantic_graph.md's "Classifier scope: live map view, not per tile". Also whenever the
    // 'detections' source finishes loading a batch of tiles (debounced -- a raster source fires
    // "sourcedata" once per tile, so a burst of N tiles finishing would otherwise mean N back-to-back
    // sends) so a detection that was still queued when moveend last fired can flip the result without
    // requiring another pan/zoom. Deliberately *not* 'idle': confirmed by direct testing that it can
    // fail to fire at all even once every tile request has genuinely completed (reproduced in headless
    // Chromium -- 'load' and every network request settled, 'idle' never did) -- too fragile a signal
    // to be the only thing keeping the live view in sync with what's actually been detected.
    // Set right before sending the *trimmed* tile list below, cleared either by the matching
    // follow-up firing (see onResult below) or by movestart (see its own handler) -- guards against
    // treating some later, unrelated result (e.g. movestart's own empty-tiles cancel response) as
    // "the trimmed stage just finished, send the full one now."
    let pendingFullFollowUp = false

    const socket = new ExtentSocket((result) => {
      setSites(result)
      const boundariesSource = map.getSource('site-boundaries') as maplibregl.GeoJSONSource | undefined
      const labelsSource = map.getSource('site-labels') as maplibregl.GeoJSONSource | undefined
      boundariesSource?.setData(result)
      labelsSource?.setData(labelsFrom(result))

      // Two-stage load: the trimmed request above gets the fast, most-likely-relevant tiles drawn
      // first; once that's back, follow up with the *untrimmed* full viewport so the skipped edge
      // margin still gets covered, just a little later rather than never. Cheap to do this way
      // instead of computing just the missing outer ring -- the inner tiles this re-requests are
      // already in tile_server's cache (TILE_CACHE_CAPACITY), so only the genuinely new margin
      // tiles cost any real inference time.
      if (pendingFullFollowUp) {
        pendingFullFollowUp = false
        const fullTiles = visibleDetectZoomTiles(map, 0)
        if (fullTiles.length > 0) socket.send(DETECT_ZOOM, fullTiles)
      }
    })

    const updateExtent = () => {
      if (map.getZoom() < MIN_DETECT_ZOOM) return
      const tiles = visibleDetectZoomTiles(map)
      if (tiles.length === 0) return
      pendingFullFollowUp = true
      socket.send(DETECT_ZOOM, tiles)
    }

    // moveend fires once a single gesture (or its momentum) has fully settled, but several
    // separate short gestures in a row (e.g. a few quick individual pans) each fire their own --
    // a short debounce here means only the last one in a quick burst actually triggers a request,
    // instead of firing (and then immediately cancelling, via movestart on the next one) a full
    // two-stage load for a view the user was never actually going to stay on.
    let moveEndDebounceTimer: ReturnType<typeof setTimeout> | undefined
    let sourcedataDebounceTimer: ReturnType<typeof setTimeout> | undefined
    map.on('load', updateExtent) // first load: no burst to debounce, fire immediately
    map.on('moveend', () => {
      clearTimeout(moveEndDebounceTimer)
      moveEndDebounceTimer = setTimeout(updateExtent, 300)
    })

    // As soon as a new gesture starts, tell the server to drop whatever it queued for the view
    // we're about to leave -- ws_server.py already cancels the in-flight classify task and prunes
    // tile_server's pending queue on every new extent report (see ws_server.py's ws_extent
    // docstring), but until now that only happened once the *next* moveend arrived, so a stale
    // batch kept getting worked on for the entire duration of the pan/zoom gesture even though it
    // was already guaranteed to be discarded. An empty-tiles report reuses that exact same
    // cancel/prune path for free -- current_tiles is empty so nothing new gets queued, but the
    // cancel-then-prune sequence still runs immediately. Jobs a worker had *already* started
    // can't be cheaply cancelled either way (see DetectionQueue.clear_pending()'s docstring) --
    // this only stops queued-but-not-yet-started work from piling up further, it doesn't abort
    // GPU calls already in flight. Also clears pendingFullFollowUp -- this message's own response
    // will reach onResult like any other, and without clearing the flag first it would be
    // misread as "the trimmed stage just finished," firing a full-coverage follow-up for a view
    // the user is already leaving. Also clears both debounce timers below -- a gesture starting
    // means any not-yet-fired debounced updateExtent() call is for a view already being left, same
    // reasoning as the empty-tiles send itself.
    map.on('movestart', () => {
      pendingFullFollowUp = false
      clearTimeout(moveEndDebounceTimer)
      clearTimeout(sourcedataDebounceTimer)
      socket.send(DETECT_ZOOM, [])
    })

    map.on('sourcedata', (e) => {
      if (e.sourceId !== 'detections' || !e.isSourceLoaded) return
      clearTimeout(sourcedataDebounceTimer)
      sourcedataDebounceTimer = setTimeout(updateExtent, 300)
    })

    return () => {
      clearTimeout(moveEndDebounceTimer)
      clearTimeout(sourcedataDebounceTimer)
      socket.close()
      map.remove()
    }
  }, [])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      {zoom < MIN_DETECT_ZOOM && (
        <div
          style={{
            position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)',
            background: 'rgba(0,0,0,0.7)', color: '#fff', padding: '6px 12px',
            borderRadius: 6, fontSize: 13, pointerEvents: 'none',
          }}
        >
          Zoom in to zoom {MIN_DETECT_ZOOM}+ to run detection
        </div>
      )}
      {sites.features.length > 0 && (
        <div
          style={{
            position: 'absolute', top: 12, right: 12, zIndex: 1,
            background: 'rgba(20,20,20,0.82)', color: '#fff', fontSize: 12,
            fontFamily: 'ui-monospace, monospace', borderRadius: 8, padding: '10px 14px',
            minWidth: 200, lineHeight: 1.6,
          }}
        >
          {sites.features.map((f, i) => (
            <div key={i} style={{ marginBottom: i < sites.features.length - 1 ? 8 : 0 }}>
              <div style={{ fontWeight: 'bold' }}>{f.properties.site}</div>
              <div>coverage: {(f.properties.type_coverage_ratio * 100).toFixed(0)}%</div>
              <div>components: {f.properties.component_count}</div>
              <div style={{ opacity: 0.8 }}>{f.properties.matched_types.join(', ')}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
