import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import {
  EMPTY_DETECTIONS, EMPTY_FEATURE_COLLECTION, INITIAL_ZOOM, type SiteFeatureCollection, type SiteFeatureProperties,
} from './api'
import { classColorExpression } from './classColors'
import { extentSocket } from './socket'
import {
  gestureStarted, layersPainted,
  mapLoaded as mapLoadedAction, reset, type RootState, siteCleared, siteProcessingStarted,
  useAppDispatch, useAppSelector, type Viewport, viewportSettled, zoomChanged,
} from './store'

const MIN_DETECT_ZOOM = 16
const DETECT_ZOOM = 17
const MIN_VISIBLE_ZOOM = 12
const REPAINT_INTERVAL_MS = 350

function formatSiteName(site: string): string {
  return site.replace(/_/g, ' ')
}

const INITIAL_CENTER: [number, number] = [0, 20]

function lonLatToTile(lon: number, lat: number, z: number): [number, number] {
  const n = 2 ** z
  const x = Math.floor(((lon + 180) / 360) * n)
  const latRad = (lat * Math.PI) / 180
  const y = Math.floor(((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n)
  return [x, y]
}

function intersects(v: Viewport, bbox: [number, number, number, number]): boolean {
  const [west, south, east, north] = bbox
  return v.west < east && v.east > west && v.south < north && v.north > south
}

function remainingSeconds(
  progress: { done: number; total: number; startedAt: number; updatedAt: number }, now: number,
): number | null {
  if (progress.done < 1 || progress.startedAt === 0) return null
  const perTileMs = (progress.updatedAt - progress.startedAt) / progress.done
  const remainingAtUpdate = perTileMs * (progress.total - progress.done)
  const elapsedSinceUpdate = now - progress.updatedAt
  return (remainingAtUpdate - elapsedSinceUpdate) / 1000
}

function currentViewport(map: maplibregl.Map): Viewport {
  const bounds = map.getBounds()
  return {
    zoom: map.getZoom(),
    west: bounds.getWest(),
    east: bounds.getEast(),
    south: bounds.getSouth(),
    north: bounds.getNorth(),
  }
}

function tilesForViewport(viewport: Viewport): { x: number; y: number }[] {
  const { west, east, south, north } = viewport
  const [xMin, yMin] = lonLatToTile(west, north, DETECT_ZOOM)
  const [xMax, yMax] = lonLatToTile(east, south, DETECT_ZOOM)
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

function labelsFrom(fc: SiteFeatureCollection): LabelFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: fc.features.map((f) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [f.properties.label_lon, f.properties.label_lat] },
      properties: { ...f.properties, site: formatSiteName(f.properties.site) },
    })),
  }
}

export default function Map() {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const dispatch = useAppDispatch()

  const zoom = useAppSelector((s: RootState) => s.map.zoom)
  const isMapLoaded = useAppSelector((s: RootState) => s.map.mapLoaded)
  const sites = useAppSelector((s: RootState) => s.map.sites)
  const readyGeneration = useAppSelector((s: RootState) => s.map.readyGeneration)
  const paintedGeneration = useAppSelector((s: RootState) => s.map.paintedGeneration)
  const viewport = useAppSelector((s: RootState) => s.map.viewport)
  const gestureActive = useAppSelector((s: RootState) => s.map.gestureActive)
  const flyTo = useAppSelector((s: RootState) => s.map.flyTo)
  const detections = useAppSelector((s: RootState) => s.map.detections)
  const selectedSite = useAppSelector((s: RootState) => s.map.selectedSite)
  const sitePhase = useAppSelector((s: RootState) => s.map.sitePhase)
  const siteProgress = useAppSelector((s: RootState) => s.map.siteProgress)
  const siteBusy = sitePhase === 'landing' || sitePhase === 'processing'
  const [now, setNow] = useState(() => Date.now())
  const siteSeenInViewRef = useRef<string | null>(null)
  const [hasRoamed, setHasRoamed] = useState(false)
  const [arrivedGeneration, setArrivedGeneration] = useState(0)
  const lastPaintAtRef = useRef(0)

  useEffect(() => {
    if (sitePhase !== 'processing') return
    const id = setInterval(() => setNow(Date.now()), 250)
    return () => clearInterval(id)
  }, [sitePhase])

  useEffect(() => {
    if (!containerRef.current) return

    const tileDebug0 = performance.now()
    const tileDebug = (msg: string, extra?: Record<string, unknown>) =>
      console.log(`[tile-debug +${(performance.now() - tileDebug0).toFixed(0)}ms] ${msg}`, extra ?? '')

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
        sources: {
          basemap: {
            type: 'raster',
            tiles: ['/api/tile/{z}/{x}/{y}'],
            tileSize: 512,
            maxzoom: DETECT_ZOOM,
            attribution: '© Mapbox',
          },
        },
        layers: [
          { id: 'basemap', type: 'raster', source: 'basemap' },
        ],
      },
      center: INITIAL_CENTER,
      zoom: INITIAL_ZOOM,
    })
    mapRef.current = map
    tileDebug('map constructed -- basemap/detections sources+layers declared in initial style')
    map.addControl(new maplibregl.NavigationControl())
    map.on('zoom', () => dispatch(zoomChanged(map.getZoom())))

    map.on('sourcedata', (e) => {
      if (e.sourceId !== 'basemap') return
      tileDebug(`sourcedata [${e.sourceId}]`, {
        isSourceLoaded: e.isSourceLoaded,
        dataType: e.dataType,
        sourceDataType: (e as unknown as { sourceDataType?: string }).sourceDataType,
      })
    })
    map.on('movestart', () => tileDebug('movestart'))
    map.on('moveend', () => tileDebug('moveend'))
    map.on('idle', () => tileDebug('idle'))

    let lastRenderLogAt = 0
    map.on('render', () => {
      const now = performance.now()
      if (now - lastRenderLogAt > 2000) {
        lastRenderLogAt = now
        tileDebug('render (throttled, logged at most 1/2s)')
      }
    })

    ;(window as typeof window & { checkTile?: (x: number, y: number) => Promise<void> }).checkTile =
      async (x: number, y: number) => {
        const res = await fetch(`/api/detections/${DETECT_ZOOM}/${x}/${y}`, { cache: 'no-store' })
        const bytes = await res.arrayBuffer()
        tileDebug(`checkTile(${x},${y})`, { status: res.status, bytes: bytes.byteLength })
      }

    ;(window as typeof window & { map?: maplibregl.Map }).map = map

    map.on('load', () => {
      tileDebug('map "load" fired')
      dispatch(mapLoadedAction())
      dispatch(viewportSettled(currentViewport(map)))
    })

    let moveEndDebounceTimer: ReturnType<typeof setTimeout> | undefined
    map.on('moveend', () => {
      clearTimeout(moveEndDebounceTimer)
      moveEndDebounceTimer = setTimeout(() => dispatch(viewportSettled(currentViewport(map))), 300)
    })

    map.on('movestart', (e) => {
      clearTimeout(moveEndDebounceTimer)
      if ((e as { originalEvent?: unknown }).originalEvent) setHasRoamed(true)
      dispatch(gestureStarted())
    })

    return () => {
      clearTimeout(moveEndDebounceTimer)
      map.remove()
      mapRef.current = null
      dispatch(reset())
    }
  }, [dispatch])

  useEffect(() => {
    if (!gestureActive || siteBusy) return
    extentSocket.send(DETECT_ZOOM, [])
  }, [gestureActive, siteBusy])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !isMapLoaded || !flyTo) return
    const [west, south, east, north] = flyTo.bbox
    const camera = map.cameraForBounds([[west, south], [east, north]], { padding: 40, maxZoom: DETECT_ZOOM })
    if (!camera) return
    const { generation } = flyTo
    const onArrive = () => setArrivedGeneration(generation)
    map.once('moveend', onArrive)
    map.flyTo({ ...camera, duration: flyTo.durationMs, essential: true })
    return () => {
      map.off('moveend', onArrive)
    }
  }, [isMapLoaded, flyTo])

  useEffect(() => {
    if (!selectedSite) return
    if (sitePhase === 'landing') {
      if (flyTo && arrivedGeneration === flyTo.generation) {
        extentSocket.sendSite(selectedSite.id)
        dispatch(siteProcessingStarted())
      }
      return
    }
    if (sitePhase !== 'done' || !viewport) return
    if (intersects(viewport, selectedSite.bbox)) {
      siteSeenInViewRef.current = selectedSite.id
    } else if (siteSeenInViewRef.current === selectedSite.id) {
      dispatch(siteCleared())
    }
  }, [dispatch, viewport, selectedSite, sitePhase, flyTo, arrivedGeneration])

  useEffect(() => {
    if (siteBusy || !viewport || viewport.zoom < MIN_DETECT_ZOOM) return
    const tiles = tilesForViewport(viewport)
    if (tiles.length > 0) extentSocket.send(DETECT_ZOOM, tiles)
  }, [viewport, siteBusy])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const handlers = [map.dragPan, map.scrollZoom, map.keyboard, map.doubleClickZoom, map.touchZoomRotate, map.boxZoom]
    for (const handler of handlers) {
      if (siteBusy) handler.disable()
      else handler.enable()
    }
  }, [siteBusy])

  useEffect(() => {
    const map = mapRef.current
    if (!isMapLoaded || !map) return
    map.addSource('site-boundaries', { type: 'geojson', data: EMPTY_FEATURE_COLLECTION })
    map.addLayer({
      id: 'site-fill', type: 'fill', source: 'site-boundaries', minzoom: MIN_VISIBLE_ZOOM,
      paint: { 'fill-color': '#ffee00', 'fill-opacity': 0.15 },
    })
    map.addLayer({
      id: 'site-outline', type: 'line', source: 'site-boundaries', minzoom: MIN_VISIBLE_ZOOM,
      paint: { 'line-color': '#ff00aa', 'line-width': 4 },
    })
    map.addSource('detections', { type: 'geojson', data: EMPTY_DETECTIONS })
    map.addLayer({
      id: 'detection-outline-weak', type: 'line', source: 'detections',
      filter: ['==', ['get', 'qualifies'], false],
      paint: {
        'line-color': classColorExpression() as maplibregl.ExpressionSpecification,
        'line-width': 2, 'line-dasharray': [2, 2],
      },
    })
    map.addLayer({
      id: 'detection-outline', type: 'line', source: 'detections',
      filter: ['!=', ['get', 'qualifies'], false],
      paint: { 'line-color': classColorExpression() as maplibregl.ExpressionSpecification, 'line-width': 2 },
    })
    map.addLayer({
      id: 'detection-label', type: 'symbol', source: 'detections', minzoom: MIN_DETECT_ZOOM,
      layout: {
        'text-field': ['get', 'label'], 'text-size': 11, 'text-font': ['Open Sans Semibold'],
        'text-anchor': 'top', 'text-offset': [0, 0.4],
      },
      paint: {
        'text-color': '#101010',
        'text-halo-color': classColorExpression() as maplibregl.ExpressionSpecification,
        'text-halo-width': 1.6,
      },
    })
    map.addSource('site-labels', { type: 'geojson', data: labelsFrom(EMPTY_FEATURE_COLLECTION) })
    map.addLayer({
      id: 'site-label', type: 'symbol', source: 'site-labels', minzoom: MIN_VISIBLE_ZOOM,
      layout: { 'text-field': ['get', 'site'], 'text-size': 16, 'text-font': ['Open Sans Semibold'] },
      paint: { 'text-color': '#fff', 'text-halo-color': '#000', 'text-halo-width': 1.5 },
    })
  }, [isMapLoaded])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !isMapLoaded || readyGeneration === paintedGeneration) return

    const paint = () => {
      const boundariesSource = map.getSource('site-boundaries') as maplibregl.GeoJSONSource | undefined
      const labelsSource = map.getSource('site-labels') as maplibregl.GeoJSONSource | undefined
      boundariesSource?.setData(sites)
      labelsSource?.setData(labelsFrom(sites))
      const detectionsSource = map.getSource('detections') as maplibregl.GeoJSONSource | undefined
      detectionsSource?.setData(detections)
      lastPaintAtRef.current = performance.now()
      dispatch(layersPainted(readyGeneration))
    }

    const sinceLastPaint = performance.now() - lastPaintAtRef.current
    if (sinceLastPaint >= REPAINT_INTERVAL_MS) {
      paint()
      return
    }
    const timer = setTimeout(paint, REPAINT_INTERVAL_MS - sinceLastPaint)
    return () => clearTimeout(timer)
  }, [dispatch, isMapLoaded, readyGeneration, paintedGeneration, sites, detections])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      {sitePhase === 'processing' && (
        <div
          style={{
            position: 'absolute', inset: 0, zIndex: 2, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 18, background: 'rgba(0,0,0,0.35)',
            color: '#fff', fontFamily: 'ui-monospace, monospace', cursor: 'wait',
          }}
        >
          <div
            style={{
              width: 96, height: 96, borderRadius: '50%', border: '10px solid rgba(255,255,255,0.25)',
              borderTopColor: '#ff00aa', animation: 'spin 1s linear infinite',
            }}
          />
          <div style={{ fontSize: 20, fontWeight: 'bold', textShadow: '0 1px 4px #000' }}>processing</div>
          <div style={{ fontSize: 16, opacity: 0.85, textShadow: '0 1px 4px #000' }}>
            {(() => {
              const left = remainingSeconds(siteProgress, now)
              if (left === null) return 'estimating…'
              return left < 1.5 ? 'almost done…' : `about ${Math.ceil(left)} s left`
            })()}
          </div>
        </div>
      )}
      {zoom < MIN_DETECT_ZOOM && !siteBusy && !selectedSite && hasRoamed && (
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
            position: 'absolute', top: 12, right: 56, zIndex: 1,
            background: 'rgba(20,20,20,0.82)', color: '#fff', fontSize: 12,
            fontFamily: 'ui-monospace, monospace', borderRadius: 8, padding: '10px 14px',
            minWidth: 200, lineHeight: 1.6,
          }}
        >
          {sites.features.map((f, i) => (
            <div key={i} style={{ marginBottom: i < sites.features.length - 1 ? 8 : 0 }}>
              <div style={{ fontWeight: 'bold' }}>{formatSiteName(f.properties.site)}</div>
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
