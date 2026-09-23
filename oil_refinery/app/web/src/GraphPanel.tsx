import type { CSSProperties } from 'react'
import type { ComponentSummary } from './api'
import { classColor } from './classColors'
import { type RootState, useAppSelector } from './store'

const GREY = '#3a3a3a'
const YELLOW = '#d9a400'
const GREEN = '#2e9e4f'

const PLACEHOLDER: ComponentSummary[] = ['storage tank', 'fan-unit', 'distillation-column'].map((component) => ({
  component, min_confidence: 0, min_count: 1, count: 0, max_confidence: null, satisfied: false,
  counts_groups: component === 'fan-unit',
}))

function childColor(c: ComponentSummary): string {
  if (c.satisfied) return GREEN
  if (c.count > 0) return YELLOW
  return GREY
}

function node(color: string, wide: boolean, outline?: string): CSSProperties {
  return {
    background: color, color: '#fff', borderRadius: 8,
    padding: wide ? '8px 18px' : '5px 9px',
    borderStyle: 'solid', borderWidth: outline ? 2 : 0, borderColor: outline ?? 'transparent',
    fontWeight: wide ? 'bold' : 'normal', fontSize: wide ? 14 : 12, textAlign: 'center',
    minWidth: wide ? 140 : 96, boxShadow: '0 1px 4px rgba(0,0,0,0.5)', transition: 'background 300ms',
  }
}

const connector: CSSProperties = { width: 2, height: 14, background: '#777' }

export default function GraphPanel() {
  const graph = useAppSelector((s: RootState) => s.map.graph)
  const selectedSite = useAppSelector((s: RootState) => s.map.selectedSite)
  const sitePhase = useAppSelector((s: RootState) => s.map.sitePhase)

  const hasCounts = (graph?.components ?? []).some((c) => c.count > 0)
  if (!selectedSite && !hasCounts) return null

  const components = graph && graph.components.length > 0 ? graph.components : PLACEHOLDER
  const identified = graph?.identified ?? false

  return (
    <div
      data-tour="graph"
      style={{
        position: 'absolute', bottom: 12, left: 12, zIndex: 1,
        background: 'rgba(20,20,20,0.88)', color: '#fff', fontFamily: 'ui-monospace, monospace',
        borderRadius: 10, padding: '12px 16px', display: 'flex', flexDirection: 'column', alignItems: 'center',
      }}
    >
      <div style={{ fontSize: 11, opacity: 0.6, alignSelf: 'flex-start', marginBottom: 8 }}>
        {selectedSite
          ? `${selectedSite.label}${sitePhase === 'done' ? '' : ' · processing'}`
          : 'current view'}
      </div>
      <div style={node(identified ? GREEN : GREY, true)}>oil refinery</div>
      <div style={connector} />
      <div style={{ width: 'calc(100% - 108px)', height: 2, background: '#777' }} />
      <div style={{ display: 'flex', gap: 12 }}>
        {components.map((c) => (
          <div key={c.component} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={connector} />
            <div style={node(childColor(c), false, classColor(c.component))}>
              <div>{c.component}</div>
              <div style={{ fontSize: 11, opacity: 0.85 }}>
                {c.count}{c.min_count > 1 ? ` / ${c.min_count}` : ''}
                {c.counts_groups ? ` group${c.count === 1 ? '' : 's'}` : ''}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
