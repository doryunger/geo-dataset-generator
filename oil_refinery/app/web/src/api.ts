export interface Stats {
  processed_total: number
  dropped_total: number
  cache_hits: number
  last_inference_ms: number | null
  avg_inference_ms: number | null
  queue_depth: number
  in_flight: number
  cached_tiles: number
  device: string
  min_detect_zoom: number
}

export async function fetchStats(): Promise<Stats> {
  const res = await fetch('/api/stats')
  if (!res.ok) throw new Error(`GET /api/stats failed: ${res.status}`)
  return res.json()
}

export interface SiteFeatureProperties {
  site: string
  matched_types: string[]
  type_coverage_ratio: number
  component_count: number
  label_lon: number
  label_lat: number
}

export interface SiteFeatureCollection {
  type: 'FeatureCollection'
  features: {
    type: 'Feature'
    geometry: { type: 'Polygon'; coordinates: number[][][] }
    properties: SiteFeatureProperties
  }[]
}

// Site-level results (identified-site boundaries) travel over a websocket, not a plain request --
// see server.py/ws_server.py's docstrings for why: a site spans the whole live view, not one tile,
// and isn't triggered by any single tile request the way /api/tile or /api/detections are. Send the
// map's current live view (see oil_refinery/semantic_graph.md's "Classifier scope: live map view,
// not per tile") via send(); onResult fires once per send() with a single, complete result -- the
// server waits for the whole reported batch to finish processing before classifying and replying
// (see ws_server.py's classify_extent()), no partial/early results. Always just replace whatever's
// currently shown with the latest result, never try to merge or accumulate across calls.
export class ExtentSocket {
  private ws: WebSocket | null = null
  private closed = false
  private onResult: (result: SiteFeatureCollection) => void

  constructor(onResult: (result: SiteFeatureCollection) => void) {
    this.onResult = onResult
    this.connect()
  }

  private connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    this.ws = new WebSocket(`${protocol}://${window.location.host}/ws/extent`)
    this.ws.onmessage = (event) => {
      try {
        this.onResult(JSON.parse(event.data))
      } catch (err) {
        console.error('ExtentSocket: failed to parse message from server', err)
      }
    }
    this.ws.onclose = () => {
      if (!this.closed) setTimeout(() => this.connect(), 1000) // transient drop -- reconnect
    }
  }

  send(zoom: number, tiles: { x: number; y: number }[]) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ zoom, tiles }))
    }
  }

  close() {
    this.closed = true
    this.ws?.close()
  }
}
