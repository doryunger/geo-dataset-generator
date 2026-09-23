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
  warm: boolean
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

export interface Site {
  id: string
  name: string
  label: string
  kind: 'refinery' | 'look-alike'
  type: string
  bbox: [number, number, number, number]
  tiles: number
}

export interface ComponentSummary {
  component: string
  min_confidence: number
  min_count: number
  count: number
  counts_groups?: boolean
  member_count?: number
  max_confidence: number | null
  satisfied: boolean
}

export interface DetectionFeatureCollection {
  type: 'FeatureCollection'
  features: {
    type: 'Feature'
    geometry: { type: 'Polygon'; coordinates: number[][][] }
    properties: { tile: string; class_name: string; confidence: number; label: string }
  }[]
}

export const EMPTY_DETECTIONS: DetectionFeatureCollection = { type: 'FeatureCollection', features: [] }

export interface ResultMessage {
  type: 'extent' | 'extent_tile' | 'site_tile' | 'site_done'
  sites?: SiteFeatureCollection
  detections?: DetectionFeatureCollection
  components?: ComponentSummary[]
  site?: string
  tile?: string
  done?: number
  total?: number
}

export async function fetchSites(): Promise<Site[]> {
  const res = await fetch('/api/sites')
  if (!res.ok) throw new Error(`GET /api/sites failed: ${res.status}`)
  return res.json()
}

export const INITIAL_ZOOM = 1

export interface ExtentSocketHandlers {
  onServerReady: () => void
  onResult: (result: ResultMessage) => void
}

export class ExtentSocket {
  private ws: WebSocket | null = null
  private closed = false
  private readonly handlers: ExtentSocketHandlers
  private readonly sessionId = crypto.randomUUID()

  constructor(handlers: ExtentSocketHandlers) {
    this.handlers = handlers
    this.connect()
  }

  private connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    this.ws = new WebSocket(`${protocol}://${window.location.host}/ws/extent?session=${this.sessionId}`)
    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'server_ready') {
          this.handlers.onServerReady()
        } else {
          this.handlers.onResult(data)
        }
      } catch (err) {
        console.error('ExtentSocket: failed to parse message from server', err)
      }
    }
    this.ws.onclose = () => {
      if (!this.closed) setTimeout(() => this.connect(), 1000)
    }
  }

  sendSite(id: string) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ site: id }))
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
