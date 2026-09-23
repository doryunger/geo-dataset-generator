import GraphPanel from './GraphPanel'
import Map from './Map'
import SitesPanel from './SitesPanel'
import StatsOverlay from './StatsOverlay'
import { type RootState, useAppSelector } from './store'

const SHOW_DEBUG = new URLSearchParams(window.location.search).has('debug')

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
