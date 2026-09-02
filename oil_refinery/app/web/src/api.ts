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
