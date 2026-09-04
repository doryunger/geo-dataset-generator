import { useEffect, useRef } from 'react'
import * as maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { EMPTY_FEATURE_COLLECTION, ExtentSocket, INITIAL_ZOOM, type SiteFeatureCollection, type SiteFeatureProperties } from './api'
import {
  extentResultReceived, fullFollowUpSent, gestureStarted, layersPainted,
  mapLoaded as mapLoadedAction, reset, type RootState, useAppDispatch, useAppSelector,
  type Viewport, viewportSettled, zoomChanged,
} from './store'

const MIN_DETECT_ZOOM = 16
const DETECT_ZOOM = 17
const MAX_AREA_TRIM = 0.3
const MIN_VISIBLE_ZOOM = 12

function formatSiteName(site: string): string {
  return site.replace(/_/g, ' ')
}

function centerWaveTrim(displayedZoom: number): number {
  const maxGap = DETECT_ZOOM - MIN_DETECT_ZOOM
  if (maxGap <= 0) return 0
  const gap = DETECT_ZOOM - displayedZoom
  const areaTrim = Math.min(MAX_AREA_TRIM, Math.max(0, MAX_AREA_TRIM * (gap / maxGap)))
  return 1 - Math.sqrt(1 - areaTrim)
}

const INITIAL_CENTER: [number, number] = [9.9517431, 53.4770211]

function lonLatToTile(lon: number, lat: number, z: number): [number, number] {
  const n = 2 ** z
  const x = Math.floor(((lon + 180) / 360) * n)
  const latRad = (lat * Math.PI) / 180
  const y = Math.floor(((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n)
  return [x, y]
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

function tilesForViewport(
  viewport: Viewport, trimFraction: number = centerWaveTrim(viewport.zoom),
): { x: number; y: number }[] {
  const { west, east, south, north } = viewport
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
  const socketRef = useRef<ExtentSocket | null>(null)
  const lastFollowUpGenRef = useRef(0)
  const dispatch = useAppDispatch()

  const zoom = useAppSelector((s: RootState) => s.map.zoom)
  const isMapLoaded = useAppSelector((s: RootState) => s.map.mapLoaded)
  const sites = useAppSelector((s: RootState) => s.map.sites)
  const readyGeneration = useAppSelector((s: RootState) => s.map.readyGeneration)
  const paintedGeneration = useAppSelector((s: RootState) => s.map.paintedGeneration)
  const viewport = useAppSelector((s: RootState) => s.map.viewport)
  const gestureActive = useAppSelector((s: RootState) => s.map.gestureActive)
  const pendingFullFollowUp = useAppSelector((s: RootState) => s.map.pendingFullFollowUp)

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
          detections: {
            type: 'raster',
            tiles: ['/api/detections/{z}/{x}/{y}'],
            tileSize: 512,
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
    mapRef.current = map
    tileDebug('map constructed -- basemap/detections sources+layers declared in initial style')
    map.addControl(new maplibregl.NavigationControl())
    map.on('zoom', () => dispatch(zoomChanged(map.getZoom())))

    map.on('sourcedata', (e) => {
      if (e.sourceId !== 'basemap' && e.sourceId !== 'detections') return
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

    const socket = new ExtentSocket((result) => {
      tileDebug('onResult', { siteCount: result.features.length })
      dispatch(extentResultReceived(result))
    })
    socketRef.current = socket

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

    map.on('movestart', () => {
      clearTimeout(moveEndDebounceTimer)
      dispatch(gestureStarted())
    })

    return () => {
      clearTimeout(moveEndDebounceTimer)
      socket.close()
      socketRef.current = null
      map.remove()
      mapRef.current = null
      lastFollowUpGenRef.current = 0
      dispatch(reset())
    }
  }, [dispatch])

  useEffect(() => {
    if (!gestureActive) return
    socketRef.current?.send(DETECT_ZOOM, [])
  }, [gestureActive])

  useEffect(() => {
    const socket = socketRef.current
    if (!socket || !viewport || viewport.zoom < MIN_DETECT_ZOOM) return
    const tiles = tilesForViewport(viewport)
    if (tiles.length > 0) socket.send(DETECT_ZOOM, tiles)
  }, [viewport])

  useEffect(() => {
    const socket = socketRef.current
    if (!pendingFullFollowUp || !socket || !viewport) return
    if (readyGeneration === lastFollowUpGenRef.current) return
    lastFollowUpGenRef.current = readyGeneration
    dispatch(fullFollowUpSent())
    if (viewport.zoom < MIN_DETECT_ZOOM) return
    const centerKeys = new Set(tilesForViewport(viewport).map(({ x, y }) => `${x},${y}`))
    const peripheralTiles = tilesForViewport(viewport, 0).filter(({ x, y }) => !centerKeys.has(`${x},${y}`))
    if (peripheralTiles.length > 0) socket.send(DETECT_ZOOM, peripheralTiles)
  }, [dispatch, readyGeneration, pendingFullFollowUp, viewport])

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

    const boundariesSource = map.getSource('site-boundaries') as maplibregl.GeoJSONSource | undefined
    const labelsSource = map.getSource('site-labels') as maplibregl.GeoJSONSource | undefined
    boundariesSource?.setData(sites)
    labelsSource?.setData(labelsFrom(sites))

    const detectionsSource = map.getSource('detections') as maplibregl.RasterTileSource | undefined
    detectionsSource?.setTiles(['/api/detections/{z}/{x}/{y}'])

    map.triggerRepaint()

    dispatch(layersPainted(readyGeneration))
  }, [dispatch, isMapLoaded, readyGeneration, paintedGeneration, sites])

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
