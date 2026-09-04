import Map from './Map'
import StatsOverlay from './StatsOverlay'
import { type RootState, useAppSelector } from './store'

export default function App() {
  const serverReady = useAppSelector((s: RootState) => s.connection.serverReady)

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh' }}>
      {serverReady ? (
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
