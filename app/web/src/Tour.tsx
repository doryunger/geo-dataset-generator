import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import type { DetectionFeatureCollection, SiteFeatureCollection } from './api'
import { mapHandle } from './mapHandle'
import { type RootState, useAppSelector } from './store'

const START_DELAY_MS = 800
const ZOOM_FLIGHT_MS = 2500
const RETURN_FLIGHT_MS = 1500
const CLUSTER_RADIUS_M = 80
const DETECTION_MAX_ZOOM = 18
const HIGHLIGHT_PAD = 8
const CARD_WIDTH = 320
const CARD_GAP = 16
const EDGE = 12
const ACCENT = '#ffd400'

type Target = { kind: 'dom'; id: string } | { kind: 'site' } | { kind: 'detections' }

interface Step {
  title: string
  body: string
  target: Target
  nextLabel?: string
}

interface Rect {
  left: number
  top: number
  right: number
  bottom: number
}

type LngLat = [number, number]

const STEPS: Step[] = [
  {
    title: 'Sites',
    body: 'Pick a site to run. The left column holds real oil refineries, the right one look-alikes '
      + '(power stations, tank farms, ports...). Each site turns green or red once it has been run. '
      + '"Free browsing" lets you roam and detect anywhere at zoom 16+.',
    target: { kind: 'dom', id: 'sites' },
  },
  {
    title: 'Semantic graph',
    body: 'What makes a refinery: storage tanks, fan units and distillation columns. A component '
      + 'turns yellow when something is detected and green once its required count is met. The '
      + '"oil refinery" node turns green only when all three are found together.',
    target: { kind: 'dom', id: 'graph' },
  },
  {
    title: 'Site verdict',
    body: 'The site the classifier identified: which component types matched, the share of required '
      + 'types covered, and how many detected components make it up.',
    target: { kind: 'dom', id: 'site-details' },
  },
  {
    title: 'Identified refinery',
    body: 'The outlined area is built from the detections themselves, not taken from a map of the site.',
    target: { kind: 'site' },
    nextLabel: 'Zoom in',
  },
  {
    title: 'Detections',
    body: 'Each box is one detected object, coloured by class. Solid boxes passed the confidence floor '
      + 'and count towards the verdict; dashed ones are weaker hits shown for context.',
    target: { kind: 'detections' },
  },
]

function polygonPoints(fc: SiteFeatureCollection | DetectionFeatureCollection): LngLat[] {
  return fc.features.flatMap((f) => f.geometry.coordinates[0] as LngLat[])
}

function centroid(ring: number[][]): LngLat {
  const pts = ring.slice(0, -1)
  const lon = pts.reduce((a, p) => a + p[0], 0) / pts.length
  const lat = pts.reduce((a, p) => a + p[1], 0) / pts.length
  return [lon, lat]
}

function distanceM(a: LngLat, b: LngLat): number {
  const kx = 111320 * Math.cos((a[1] * Math.PI) / 180)
  return Math.hypot((a[0] - b[0]) * kx, (a[1] - b[1]) * 110540)
}

function densestCluster(detections: DetectionFeatureCollection): LngLat[] {
  const strong = detections.features.filter((f) => f.properties.qualifies !== false)
  const pool = strong.length > 0 ? strong : detections.features
  if (pool.length === 0) return []
  const centers = pool.map((f) => centroid(f.geometry.coordinates[0]))
  const neighbours = centers.map((c) => centers.filter((o) => distanceM(c, o) <= CLUSTER_RADIUS_M).length)
  const seed = centers[neighbours.indexOf(Math.max(...neighbours))]
  return pool
    .filter((_, i) => distanceM(seed, centers[i]) <= CLUSTER_RADIUS_M)
    .flatMap((f) => f.geometry.coordinates[0] as LngLat[])
}

function lngLatBounds(points: LngLat[]): [LngLat, LngLat] {
  const lons = points.map((p) => p[0])
  const lats = points.map((p) => p[1])
  return [[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]]
}

function screenRect(points: LngLat[]): Rect | null {
  const map = mapHandle.current
  if (!map || points.length === 0) return null
  const projected = points.map((p) => map.project(p))
  const xs = projected.map((p) => p.x)
  const ys = projected.map((p) => p.y)
  const box = map.getContainer().getBoundingClientRect()
  return {
    left: Math.max(box.left + Math.min(...xs), EDGE),
    top: Math.max(box.top + Math.min(...ys), EDGE),
    right: Math.min(box.left + Math.max(...xs), window.innerWidth - EDGE),
    bottom: Math.min(box.top + Math.max(...ys), window.innerHeight - EDGE),
  }
}

function domRect(id: string): Rect | null {
  const el = document.querySelector(`[data-tour="${id}"]`)
  if (!el) return null
  const { left, top, right, bottom } = el.getBoundingClientRect()
  return { left, top, right, bottom }
}

function padded(r: Rect): Rect {
  return {
    left: Math.max(r.left - HIGHLIGHT_PAD, 2),
    top: Math.max(r.top - HIGHLIGHT_PAD, 2),
    right: Math.min(r.right + HIGHLIGHT_PAD, window.innerWidth - 2),
    bottom: Math.min(r.bottom + HIGHLIGHT_PAD, window.innerHeight - 2),
  }
}

function placeCard(r: Rect, w: number, h: number): { left: number; top: number } {
  const vw = window.innerWidth
  const vh = window.innerHeight
  const clampX = (x: number) => Math.min(Math.max(x, EDGE), vw - w - EDGE)
  const clampY = (y: number) => Math.min(Math.max(y, EDGE), vh - h - EDGE)
  if (r.right + CARD_GAP + w <= vw - EDGE) return { left: r.right + CARD_GAP, top: clampY(r.top) }
  if (r.left - CARD_GAP - w >= EDGE) return { left: r.left - CARD_GAP - w, top: clampY(r.top) }
  if (r.bottom + CARD_GAP + h <= vh - EDGE) return { left: clampX(r.left), top: r.bottom + CARD_GAP }
  if (r.top - CARD_GAP - h >= EDGE) return { left: clampX(r.left), top: r.top - CARD_GAP - h }
  return { left: clampX((r.left + r.right - w) / 2), top: clampY(r.bottom - h - CARD_GAP) }
}

const blocker: CSSProperties = { position: 'fixed', inset: 0, zIndex: 1000 }

const card: CSSProperties = {
  position: 'fixed', zIndex: 1001, width: CARD_WIDTH, boxSizing: 'border-box',
  background: '#1c1c1c', color: '#eee', borderRadius: 10, padding: '14px 16px',
  fontFamily: 'ui-monospace, monospace', fontSize: 13, lineHeight: 1.5,
  boxShadow: '0 4px 18px rgba(0,0,0,0.6)', border: `1px solid ${ACCENT}`,
}

function button(primary: boolean): CSSProperties {
  return {
    background: primary ? ACCENT : 'transparent', color: primary ? '#1a1a1a' : '#bbb',
    border: `1px solid ${primary ? ACCENT : '#555'}`, borderRadius: 6, padding: '5px 12px',
    fontFamily: 'inherit', fontSize: 12, fontWeight: primary ? 'bold' : 'normal', cursor: 'pointer',
  }
}

export default function Tour() {
  const mode = useAppSelector((s: RootState) => s.map.mode)
  const sitePhase = useAppSelector((s: RootState) => s.map.sitePhase)
  const readyGeneration = useAppSelector((s: RootState) => s.map.readyGeneration)
  const paintedGeneration = useAppSelector((s: RootState) => s.map.paintedGeneration)
  const sites = useAppSelector((s: RootState) => s.map.sites)
  const detections = useAppSelector((s: RootState) => s.map.detections)
  const selectedSite = useAppSelector((s: RootState) => s.map.selectedSite)

  const [started, setStarted] = useState(false)
  const [steps, setSteps] = useState<Step[]>([])
  const [index, setIndex] = useState(-1)
  const [flying, setFlying] = useState(false)
  const [rect, setRect] = useState<Rect | null>(null)
  const [cardPos, setCardPos] = useState<{ left: number; top: number } | null>(null)
  const cardRef = useRef<HTMLDivElement>(null)
  const clusterRef = useRef<LngLat[]>([])

  const step = index >= 0 && index < steps.length ? steps[index] : null

  useEffect(() => {
    if (started || mode !== 'guided' || sitePhase !== 'done') return
    if (readyGeneration !== paintedGeneration) return
    const timer = setTimeout(() => {
      const available = STEPS.filter((s) => {
        if (s.target.kind === 'dom') return domRect(s.target.id) !== null
        if (s.target.kind === 'site') return sites.features.length > 0
        return detections.features.length > 0
      })
      setStarted(true)
      setSteps(available)
      setIndex(0)
    }, START_DELAY_MS)
    return () => clearTimeout(timer)
  }, [started, mode, sitePhase, readyGeneration, paintedGeneration, sites, detections])

  const measure = useCallback(() => {
    if (!step) return setRect(null)
    const t = step.target
    const r = t.kind === 'dom' ? domRect(t.id)
      : t.kind === 'site' ? screenRect(polygonPoints(sites))
      : screenRect(clusterRef.current)
    setRect(r ? padded(r) : null)
  }, [step, sites])

  useEffect(() => {
    if (flying) return
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [measure, flying])

  useLayoutEffect(() => {
    if (!rect || !cardRef.current) return setCardPos(null)
    const { width, height } = cardRef.current.getBoundingClientRect()
    setCardPos(placeCard(rect, width, height))
  }, [rect, index])

  const flyThen = useCallback((options: Parameters<NonNullable<typeof mapHandle.current>['flyTo']>[0], then: () => void) => {
    const map = mapHandle.current
    if (!map) return then()
    setFlying(true)
    map.once('moveend', () => {
      setFlying(false)
      then()
    })
    map.flyTo({ ...options, essential: true })
  }, [])

  const finish = useCallback(() => {
    setIndex(-1)
    const map = mapHandle.current
    if (!map || !selectedSite || !clusterRef.current.length) return
    const [west, south, east, north] = selectedSite.bbox
    const camera = map.cameraForBounds([[west, south], [east, north]], { padding: 40, maxZoom: 17 })
    if (camera) map.flyTo({ ...camera, duration: RETURN_FLIGHT_MS, essential: true })
  }, [selectedSite])

  const next = useCallback(() => {
    const upcoming = steps[index + 1]
    if (!upcoming) return finish()
    if (upcoming.target.kind !== 'detections') return setIndex(index + 1)
    const map = mapHandle.current
    clusterRef.current = densestCluster(detections)
    if (!map || clusterRef.current.length === 0) return setIndex(index + 1)
    const camera = map.cameraForBounds(lngLatBounds(clusterRef.current), {
      padding: 160, maxZoom: DETECTION_MAX_ZOOM,
    })
    if (!camera) return setIndex(index + 1)
    setRect(null)
    flyThen({ ...camera, duration: ZOOM_FLIGHT_MS }, () => setIndex(index + 1))
  }, [steps, index, detections, finish, flyThen])

  useEffect(() => {
    if (!step) return
    const onKey = (e: KeyboardEvent) => {
      if (flying) return
      if (e.key === 'Escape') finish()
      else if (e.key === 'Enter' || e.key === 'ArrowRight') next()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [step, flying, next, finish])

  if (!step) return null

  return (
    <>
      <div style={blocker} />
      {!flying && rect && (
        <div
          style={{
            position: 'fixed', zIndex: 1000, pointerEvents: 'none',
            left: rect.left, top: rect.top, width: rect.right - rect.left, height: rect.bottom - rect.top,
            border: `2px solid ${ACCENT}`, borderRadius: 10, boxSizing: 'border-box',
            boxShadow: '0 0 0 9999px rgba(0,0,0,0.65)', transition: 'all 300ms ease',
          }}
        />
      )}
      {!flying && rect && (
        <div
          ref={cardRef}
          style={{ ...card, left: cardPos?.left ?? 0, top: cardPos?.top ?? 0, visibility: cardPos ? 'visible' : 'hidden' }}
        >
          <div style={{ fontSize: 11, opacity: 0.55, marginBottom: 4 }}>{index + 1} / {steps.length}</div>
          <div style={{ fontSize: 15, fontWeight: 'bold', marginBottom: 6 }}>{step.title}</div>
          <div>{step.body}</div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 14 }}>
            <button style={button(false)} onClick={finish}>Skip</button>
            <button style={button(true)} onClick={next}>
              {index === steps.length - 1 ? 'Finish' : step.nextLabel ?? 'Next'}
            </button>
          </div>
        </div>
      )}
    </>
  )
}
