import { useEffect, useState } from 'react'
import Map from './Map'
import StatsOverlay from './StatsOverlay'
import { fetchStats } from './api'

const BACKEND_POLL_INTERVAL_MS = 1000

export default function App() {
  const [backendReady, setBackendReady] = useState(false)

  useEffect(() => {
    if (backendReady) return
    let cancelled = false
    const check = async () => {
      try {
        await fetchStats()
        if (!cancelled) setBackendReady(true)
      } catch {
      }
    }
    check()
    const id = setInterval(check, BACKEND_POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [backendReady])

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh' }}>
      {backendReady ? (
        <>
          <Map />
          <StatsOverlay />
        </>
      ) : (
        <div
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: '100%', height: '100%', background: '#111', color: '#fff',
            fontFamily: 'ui-monospace, monospace', fontSize: 14,
          }}
        >
          Waiting for backend to start…
        </div>
      )}
    </div>
  )
}
