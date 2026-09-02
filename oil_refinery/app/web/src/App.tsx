import Map from './Map'
import StatsOverlay from './StatsOverlay'

export default function App() {
  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh' }}>
      <Map />
      <StatsOverlay />
    </div>
  )
}
