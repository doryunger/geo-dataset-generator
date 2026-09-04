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
  id: string
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

export const EMPTY_FEATURE_COLLECTION: SiteFeatureCollection = { type: 'FeatureCollection', features: [] }

export const INITIAL_ZOOM = 14

export class ExtentSocket {
  private ws: WebSocket | null = null
  private closed = false
  private onResult: (result: SiteFeatureCollection) => void
  private readonly sessionId = crypto.randomUUID()

  constructor(onResult: (result: SiteFeatureCollection) => void) {
    this.onResult = onResult
    this.connect()
  }

  private connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    this.ws = new WebSocket(`${protocol}://${window.location.host}/ws/extent?session=${this.sessionId}`)
    this.ws.onmessage = (event) => {
      try {
        this.onResult(JSON.parse(event.data))
      } catch (err) {
        console.error('ExtentSocket: failed to parse message from server', err)
      }
    }
    this.ws.onclose = () => {
      if (!this.closed) setTimeout(() => this.connect(), 1000)
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
