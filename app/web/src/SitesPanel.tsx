import { Fragment, useEffect, useState, type CSSProperties } from 'react'
import { fetchSites, fetchStats, type Site } from './api'
import {
  backendWarmed, modeChanged, type RootState, siteSelected, useAppDispatch, useAppSelector,
} from './store'

const WARM_POLL_INTERVAL_MS = 1000

const VERDICT_GREEN = '#16c60c'
const VERDICT_RED = '#ff2d2d'

const HIGHLIGHT_YELLOW = '#ffd400'

const panel: CSSProperties = {
  position: 'absolute', top: 12, left: 12, zIndex: 1,
  background: 'rgba(20,20,20,0.88)', color: '#eee',
  fontFamily: 'ui-monospace, monospace', fontSize: 13,
  borderRadius: 10, padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10,
}

const grid: CSSProperties = {
  display: 'grid', gridTemplateColumns: '1fr 1fr', columnGap: 14, rowGap: 3,
}

const heading: CSSProperties = {
  fontSize: 14, fontWeight: 'bold', letterSpacing: 1.2, textTransform: 'uppercase',
  opacity: 0.85, textAlign: 'center', marginBottom: 6,
}

const subtitle: CSSProperties = {
  fontSize: 10, fontStyle: 'italic', opacity: 0.6, lineHeight: 1.2, marginTop: 1,
}

function modeButton(free: boolean, busy: boolean): CSSProperties {
  return {
    width: '100%', boxSizing: 'border-box',
    background: free ? HIGHLIGHT_YELLOW : 'transparent',
    color: free ? '#1a1a1a' : 'inherit',
    borderStyle: 'solid', borderColor: free ? HIGHLIGHT_YELLOW : '#666', borderWidth: 1,
    borderRadius: 6, padding: '6px 10px', textAlign: 'center',
    fontFamily: 'inherit', fontSize: 13, fontWeight: free ? 'bold' : 'normal',
    opacity: busy ? 0.45 : 1, cursor: busy ? 'default' : 'pointer',
  }
}

function siteButton(selected: boolean, disabled: boolean, verdict: boolean | undefined): CSSProperties {
  const accent = verdict === undefined ? null : verdict ? VERDICT_GREEN : VERDICT_RED
  return {
    display: 'flex', flexDirection: 'column', justifyContent: 'center',
    width: '100%', height: '100%', boxSizing: 'border-box', textAlign: 'center',
    background: accent ? `${accent}26` : selected ? '#3a3a3a' : 'transparent',
    color: accent ?? 'inherit',
    borderStyle: 'solid',
    borderColor: accent ?? (selected ? '#777' : 'transparent'),
    borderWidth: 1, borderRadius: 5, padding: '4px 8px',
    cursor: disabled ? 'default' : 'pointer', opacity: disabled && !selected ? 0.45 : 1,
    fontFamily: 'inherit', fontSize: 'inherit', lineHeight: 1.3, whiteSpace: 'nowrap',
  }
}

function SiteCell({ site, selectedId, disabled, showType, verdicts, onSelect }: {
  site: Site | undefined; selectedId: string | null; disabled: boolean; showType: boolean
  verdicts: Record<string, boolean>; onSelect: (site: Site) => void
}) {
  if (!site) return <div />
  return (
    <button
      style={siteButton(site.id === selectedId, disabled, verdicts[site.id])}
      disabled={disabled} onClick={() => onSelect(site)}
    >
      <div>{site.label}</div>
      {showType && <div style={subtitle}>{site.type}</div>}
    </button>
  )
}

export default function SitesPanel() {
  const dispatch = useAppDispatch()
  const selectedSite = useAppSelector((s: RootState) => s.map.selectedSite)
  const sitePhase = useAppSelector((s: RootState) => s.map.sitePhase)
  const verdicts = useAppSelector((s: RootState) => s.map.siteVerdicts)
  const mode = useAppSelector((s: RootState) => s.map.mode)
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
    if (introDone || !backendWarm || sites.length === 0 || mode !== 'guided') return
    setIntroDone(true)
    dispatch(siteSelected({ site: sites[0] }))
  }, [dispatch, introDone, backendWarm, sites, mode])

  const free = mode === 'free'
  const busy = !backendWarm || sitePhase === 'landing' || sitePhase === 'processing'
  const select = (site: Site) => dispatch(siteSelected({ site }))

  if (sites.length === 0) return null

  const refineries = sites.filter((s) => s.kind === 'refinery')
  const others = sites.filter((s) => s.kind !== 'refinery')
  const rows = Math.max(refineries.length, others.length)

  return (
    <div data-tour="sites" style={panel}>
      <div style={grid}>
        <div style={heading}>Oil refineries</div>
        <div style={heading}>Other sites</div>
        {Array.from({ length: rows }, (_, i) => (
          <Fragment key={i}>
            <SiteCell
              site={refineries[i]} selectedId={selectedSite?.id ?? null} disabled={busy || free}
              showType={false} verdicts={verdicts} onSelect={select}
            />
            <SiteCell
              site={others[i]} selectedId={selectedSite?.id ?? null} disabled={busy || free}
              showType verdicts={verdicts} onSelect={select}
            />
          </Fragment>
        ))}
      </div>
      <button
        style={modeButton(free, busy)}
        disabled={busy} onClick={() => dispatch(modeChanged(free ? 'guided' : 'free'))}
      >
        {free ? 'Back to guided tour' : 'Free browsing'}
      </button>
    </div>
  )
}
