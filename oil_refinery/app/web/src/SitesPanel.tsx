import { useEffect, useState, type CSSProperties } from 'react'
import { fetchSites, fetchStats, type Site } from './api'
import { backendWarmed, type RootState, siteSelected, useAppDispatch, useAppSelector } from './store'

const WARM_POLL_INTERVAL_MS = 1000
const INTRO_FLIGHT_MS = 15000

const VERDICT_GREEN = '#16c60c'
const VERDICT_RED = '#ff2d2d'

const panel: CSSProperties = {
  position: 'absolute', top: 12, left: 12, zIndex: 1,
  background: 'rgba(20,20,20,0.88)', color: '#eee',
  fontFamily: 'ui-monospace, monospace', fontSize: 12,
  borderRadius: 10, padding: '10px 12px', display: 'flex', gap: 18,
}

const heading: CSSProperties = {
  fontSize: 10, letterSpacing: 1, textTransform: 'uppercase', opacity: 0.55, marginBottom: 6,
}

function siteButton(selected: boolean, disabled: boolean, verdict: boolean | undefined): CSSProperties {
  const accent = verdict === undefined ? null : verdict ? VERDICT_GREEN : VERDICT_RED
  return {
    display: 'block', width: '100%', textAlign: 'left', boxSizing: 'border-box',
    background: accent ? `${accent}26` : selected ? '#3a3a3a' : 'transparent',
    color: accent ?? 'inherit',
    borderStyle: 'solid',
    borderColor: accent ?? (selected ? '#777' : 'transparent'),
    borderWidth: 1, borderRadius: 5, padding: '3px 7px', marginBottom: 2,
    cursor: disabled ? 'default' : 'pointer', opacity: disabled && !selected ? 0.45 : 1,
    fontFamily: 'inherit', fontSize: 'inherit', lineHeight: 1.5, whiteSpace: 'nowrap',
  }
}

function SiteColumn({ title, sites, selectedId, disabled, verdicts, onSelect }: {
  title: string; sites: Site[]; selectedId: string | null; disabled: boolean
  verdicts: Record<string, boolean>; onSelect: (site: Site) => void
}) {
  return (
    <div>
      <div style={heading}>{title}</div>
      {sites.map((site) => (
        <button
          key={site.id} style={siteButton(site.id === selectedId, disabled, verdicts[site.id])}
          disabled={disabled} onClick={() => onSelect(site)}
        >
          {site.label}
        </button>
      ))}
    </div>
  )
}

export default function SitesPanel() {
  const dispatch = useAppDispatch()
  const selectedSite = useAppSelector((s: RootState) => s.map.selectedSite)
  const sitePhase = useAppSelector((s: RootState) => s.map.sitePhase)
  const verdicts = useAppSelector((s: RootState) => s.map.siteVerdicts)
  const backendWarm = useAppSelector((s: RootState) => s.connection.backendWarm)
  const [sites, setSites] = useState<Site[]>([])
  const [introDone, setIntroDone] = useState(false)

  useEffect(() => {
    fetchSites().then(setSites).catch(() => setSites([]))
  }, [])

  useEffect(() => {
    if (backendWarm) return
    let cancelled = false
    const poll = async () => {
      try {
        const stats = await fetchStats()
        if (!cancelled && stats.warm) dispatch(backendWarmed())
      } catch {
        // backend still starting; the next tick tries again
      }
    }
    poll()
    const id = setInterval(poll, WARM_POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [dispatch, backendWarm])

  useEffect(() => {
    if (introDone || !backendWarm || sites.length === 0) return
    setIntroDone(true)
    dispatch(siteSelected({ site: sites[0], durationMs: INTRO_FLIGHT_MS }))
  }, [dispatch, introDone, backendWarm, sites])

  const busy = !backendWarm || sitePhase === 'landing' || sitePhase === 'processing'
  const select = (site: Site) => dispatch(siteSelected({ site }))

  if (sites.length === 0) return null

  return (
    <div style={panel}>
      <SiteColumn
        title="Oil refineries" sites={sites.filter((s) => s.kind === 'refinery')}
        selectedId={selectedSite?.id ?? null} disabled={busy} verdicts={verdicts} onSelect={select}
      />
      <SiteColumn
        title="Others" sites={sites.filter((s) => s.kind !== 'refinery')}
        selectedId={selectedSite?.id ?? null} disabled={busy} verdicts={verdicts} onSelect={select}
      />
    </div>
  )
}
