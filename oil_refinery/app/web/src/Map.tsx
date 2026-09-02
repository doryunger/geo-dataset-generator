import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

// Same Hamburg refinery site already used elsewhere in this repo's own probing
// (oil_refinery/probe_pretrained.py) -- a known-good spot with real storage tanks to look at.
const INITIAL_CENTER: [number, number] = [9.9517431, 53.4770211]
const INITIAL_ZOOM = 17

// Matches the server's MIN_DETECT_ZOOM (oil_refinery/app/server/server.py). Purely informational
// here -- the server is the actual source of truth for whether a tile gets detection applied;
// this hint just explains to the user why tiles below this zoom show no annotations.
const MIN_DETECT_ZOOM = 14

export default function Map() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [zoom, setZoom] = useState(INITIAL_ZOOM)

  useEffect(() => {
    if (!containerRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          // Two independent sources stacked as two layers, not one -- base imagery is never held
          // up by how long detection takes (it's always fetched and rendered on its own), and the
          // transparent detections overlay just pops in on top whenever its tile finishes. Both
          // point at our own backend (not Mapbox directly): MapLibre's normal tile-loading
          // behavior (figure out visible tiles, fetch, cache, abort stale fetches on pan/zoom) is
          // the entire trigger mechanism for both. See server.py's two endpoints.
          basemap: {
            type: 'raster',
            tiles: ['/api/tile/{z}/{x}/{y}'],
            tileSize: 512,
            attribution: '© Mapbox',
          },
          detections: {
            type: 'raster',
            tiles: ['/api/detections/{z}/{x}/{y}'],
            tileSize: 512,
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

    return () => map.remove()
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
          Zoom in to zoom {MIN_DETECT_ZOOM}+ to run storage-tank detection
        </div>
      )}
    </div>
  )
}
