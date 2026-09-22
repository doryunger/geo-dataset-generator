const COLORS: Record<string, string> = {
  storagetank: '#00c8ff',
  fanunit: '#ffb000',
  distillationcolumn: '#ff2bd1',
}

const FALLBACK = '#c8c8c8'

function normalize(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]/g, '')
}

export function classColor(name: string): string {
  const key = normalize(name)
  for (const [candidate, color] of Object.entries(COLORS)) {
    if (key.includes(candidate) || candidate.includes(key)) return color
  }
  return FALLBACK
}

export function classColorExpression(): unknown[] {
  return [
    'match', ['get', 'class_name'],
    ['storage tank', 'storagetank'], COLORS.storagetank,
    ['fan-unit', 'fanunit', 'fan unit'], COLORS.fanunit,
    ['distillation-column', 'distillationcolumn', 'distillation column'], COLORS.distillationcolumn,
    FALLBACK,
  ]
}
