import { useEffect, useState, type CSSProperties } from 'react'
import { fetchStats, type Stats } from './api'

const POLL_INTERVAL_MS = 1500

const row: CSSProperties = { display: 'flex', justifyContent: 'space-between', gap: 16 }

export default function StatsOverlay() {
  const [stats, setStats] = useState<Stats | null>(null)

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const s = await fetchStats()
        if (!cancelled) setStats(s)
      } catch {
      }
    }
    poll()
    const id = setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  if (!stats) return null

  const fmt = (ms: number | null) => (ms === null ? '—' : `${ms.toFixed(0)} ms`)

  return (
    <div
      style={{
        position: 'absolute', bottom: 12, left: 12, zIndex: 1,
        background: 'rgba(20,20,20,0.82)', color: '#fff', fontSize: 12, fontFamily: 'ui-monospace, monospace',
        borderRadius: 8, padding: '10px 14px', minWidth: 190, lineHeight: 1.6,
      }}
    >
      <div style={row}><span>device</span><span>{stats.device}</span></div>
      <div style={row}><span>processed</span><span>{stats.processed_total}</span></div>
      <div style={row}><span>dropped</span><span>{stats.dropped_total}</span></div>
      <div style={row}><span>cache hits</span><span>{stats.cache_hits}</span></div>
      <div style={row}><span>last tile</span><span>{fmt(stats.last_inference_ms)}</span></div>
      <div style={row}><span>avg tile</span><span>{fmt(stats.avg_inference_ms)}</span></div>
      <div style={row}><span>queue</span><span>{stats.queue_depth}</span></div>
      <div style={row}><span>in flight</span><span>{stats.in_flight}</span></div>
    </div>
  )
}
