import GraphPanel from './GraphPanel'
import Map from './Map'
import SitesPanel from './SitesPanel'
import StatsOverlay from './StatsOverlay'
import Tour from './Tour'
import { type RootState, useAppSelector } from './store'

const PARAMS = new URLSearchParams(window.location.search)
const SHOW_DEBUG = PARAMS.has('debug')
const TOUR_ENABLED = import.meta.env.VITE_TOUR === '1' || PARAMS.has('tour')

export default function App() {
  const serverReady = useAppSelector((s: RootState) => s.connection.serverReady)

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh' }}>
      {serverReady ? (
        <>
          <Map />
          <SitesPanel />
          <GraphPanel />
          {SHOW_DEBUG && <StatsOverlay />}
          {TOUR_ENABLED && <Tour />}
        </>
      ) : (
        <div
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: '100%', height: '100%', background: '#111', color: '#fff',
            fontFamily: 'ui-monospace, monospace', fontSize: 14,
          }}
        >
          Warming up…
        </div>
      )}
    </div>
  )
}
