import { useEffect, useState, type CSSProperties } from 'react'
import { fetchSites, type Site } from './api'
import { type RootState, siteSelected, useAppDispatch, useAppSelector } from './store'

const PANEL_WIDTH = 340

const panel: CSSProperties = {
  width: PANEL_WIDTH, flexShrink: 0, height: '100%', overflowY: 'auto', boxSizing: 'border-box',
  background: '#161616', color: '#eee', fontFamily: 'ui-monospace, monospace', fontSize: 13,
  padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 14,
}

const heading: CSSProperties = { fontSize: 11, letterSpacing: 1, textTransform: 'uppercase', opacity: 0.55, marginBottom: 6 }

function siteButton(selected: boolean, disabled: boolean): CSSProperties {
  return {
    display: 'block', width: '100%', textAlign: 'left', boxSizing: 'border-box',
    background: selected ? '#2d2d2d' : 'transparent', color: 'inherit', border: '1px solid',
    borderColor: selected ? '#555' : 'transparent', borderRadius: 6, padding: '6px 8px',
    cursor: disabled ? 'default' : 'pointer', opacity: disabled && !selected ? 0.5 : 1,
    fontFamily: 'inherit', fontSize: 'inherit', lineHeight: 1.4,
  }
}

function SiteList({ title, sites, selectedId, disabled, onSelect }: {
  title: string; sites: Site[]; selectedId: string | null; disabled: boolean; onSelect: (site: Site) => void
}) {
  return (
    <div>
      <div style={heading}>{title}</div>
      {sites.map((site) => (
        <button
          key={site.id} style={siteButton(site.id === selectedId, disabled)} disabled={disabled}
          onClick={() => onSelect(site)}
        >
          <div>{site.name}</div>
          <div style={{ opacity: 0.6, fontSize: 11 }}>
            {site.kind === 'look-alike' ? `${site.type} · ` : ''}{site.tiles} tiles
          </div>
        </button>
      ))}
    </div>
  )
}

export default function SitesPanel() {
  const dispatch = useAppDispatch()
  const selectedSite = useAppSelector((s: RootState) => s.map.selectedSite)
  const sitePhase = useAppSelector((s: RootState) => s.map.sitePhase)
  const [sites, setSites] = useState<Site[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchSites().then(setSites).catch((err) => setError(String(err)))
  }, [])

  const busy = sitePhase === 'landing' || sitePhase === 'processing'
  const select = (site: Site) => dispatch(siteSelected(site))

  return (
    <div style={panel}>
      <div style={{ fontSize: 15, fontWeight: 'bold' }}>Site classifier</div>
      <div style={{ opacity: 0.7, lineHeight: 1.5 }}>
        Pick a site. The map fits it, then every zoom-17 tile inside it is run through the detectors,
        live. Detections appear as tiles finish; the graph on the map fills in as components reach
        their required count. A refinery is all four components together, within 300 m.
      </div>
      <SiteList
        title="Refineries" sites={sites.filter((s) => s.kind === 'refinery')}
        selectedId={selectedSite?.id ?? null} disabled={busy} onSelect={select}
      />
      <SiteList
        title="Look-alikes" sites={sites.filter((s) => s.kind === 'look-alike')}
        selectedId={selectedSite?.id ?? null} disabled={busy} onSelect={select}
      />
      {error && <div style={{ color: '#ff6b6b' }}>{error}</div>}
    </div>
  )
}
