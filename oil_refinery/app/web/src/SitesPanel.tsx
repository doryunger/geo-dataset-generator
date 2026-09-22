import { useEffect, useState, type CSSProperties } from 'react'
import { fetchSites, type Site } from './api'
import { type RootState, siteSelected, useAppDispatch, useAppSelector } from './store'

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
  const [sites, setSites] = useState<Site[]>([])

  useEffect(() => {
    fetchSites()
      .then((loaded) => {
        setSites(loaded)
        if (loaded.length > 0) dispatch(siteSelected({ site: loaded[0], durationMs: 9000 }))
      })
      .catch(() => setSites([]))
  }, [dispatch])

  const busy = sitePhase === 'landing' || sitePhase === 'processing'
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
