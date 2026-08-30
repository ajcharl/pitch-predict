// Chart palette, kept in one place so Recharts and the stylesheet cannot drift.
//
// SVG attributes don't reliably resolve CSS custom properties across browsers,
// so charts take literal values from here. These must mirror the tokens in
// styles.css — if you change one, change both.

export const chart = {
  // Neutrals
  text: '#f4f4f5',
  textDim: 'rgba(244,244,245,0.56)',
  textFaint: 'rgba(244,244,245,0.34)',
  grid: 'rgba(255,255,255,0.07)',
  axis: 'rgba(255,255,255,0.12)',
  surface: '#16171a',
  border: 'rgba(255,255,255,0.16)',
  cursor: 'rgba(255,255,255,0.05)',

  // Accent ramp — the default for a single-series chart on this theme
  accent: '#ffffff',
  accentDim: '#a1a1aa',
  accentDeep: '#6b6b73',

  // Semantic — only where colour carries meaning (win/draw/loss)
  pos: '#3fb950',
  warn: '#d29922',
  neg: '#f85149',

  // Muted fills for reference/comparison series
  muted: 'rgba(255,255,255,0.16)',
}

export const axisTick = {
  fill: chart.textDim,
  fontSize: 11,
  fontFamily: 'JetBrains Mono',
}

export const axisTickSans = {
  fill: chart.textDim,
  fontSize: 11,
}

/** Shared Recharts <Tooltip> styling. */
export const tooltipStyle = {
  contentStyle: {
    background: chart.surface,
    border: `1px solid ${chart.border}`,
    borderRadius: 8,
    fontFamily: 'JetBrains Mono',
    fontSize: 12,
    color: chart.text,
  },
  labelStyle: { color: chart.text },
  itemStyle: { color: chart.textDim },
}
