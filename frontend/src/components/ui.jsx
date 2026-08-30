// Small shared UI primitives: loading skeletons, error/empty states, badges,
// probability bars and the animated confidence counter.

import { useEffect, useState } from 'react'

export function Skeleton({ h = 16, w = '100%', style }) {
  return <div className="skeleton" style={{ height: h, width: w, ...style }} />
}

export function SkeletonCard({ rows = 3 }) {
  return (
    <div className="card">
      <Skeleton h={11} w="34%" style={{ marginBottom: 16 }} />
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} h={13} w={`${88 - i * 13}%`} style={{ marginBottom: 10 }} />
      ))}
    </div>
  )
}

export function SkeletonTable({ rows = 8 }) {
  return (
    <div className="card">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
          <Skeleton h={17} w="16%" />
          <Skeleton h={17} w="36%" />
          <Skeleton h={17} w="16%" />
          <Skeleton h={17} w="16%" />
          <Skeleton h={17} w="16%" />
        </div>
      ))}
    </div>
  )
}

export function ErrorState({ error, onRetry }) {
  return (
    <div className="card error-state">
      <div className="msg">{error?.message || 'Something went wrong.'}</div>
      {error?.suggestions?.length > 0 && (
        <div style={{ marginBottom: 14, fontSize: 12.5 }}>
          Did you mean: <strong>{error.suggestions.join(', ')}</strong>?
        </div>
      )}
      {onRetry && (
        <button className="btn ghost sm" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  )
}

export function EmptyState({ icon = '⚽', title, hint }) {
  return (
    <div className="card empty-state">
      <div className="big">{icon}</div>
      <div style={{ color: 'var(--text)', fontWeight: 600, marginBottom: 5 }}>{title}</div>
      {hint && <div style={{ fontSize: 12.5 }}>{hint}</div>}
    </div>
  )
}

export function Loading({ label = 'Computing prediction…' }) {
  return (
    <div className="card empty-state">
      <div className="spinner" />
      <div style={{ fontSize: 13 }}>{label}</div>
    </div>
  )
}

/** Last-5 results as coloured W/D/L circles, newest first. */
export function FormBadges({ results, size }) {
  const list = results && results.length ? results : []
  if (!list.length) {
    return <span style={{ color: 'var(--text-faint)', fontSize: 12 }}>No recent matches</span>
  }
  return (
    <div className="form-badges">
      {list.map((r, i) => (
        <div
          key={i}
          className={`form-badge ${r}`}
          style={size ? { width: size, height: size, fontSize: size * 0.45 } : undefined}
          title={r === 'W' ? 'Win' : r === 'D' ? 'Draw' : 'Loss'}
        >
          {r}
        </div>
      ))}
    </div>
  )
}

/** Trust badge: 1-10, green when the model is confident. */
export function TrustBadge({ score }) {
  const cls = score >= 7 ? '' : score >= 4 ? 'mid' : 'low'
  return (
    <span className={`trust ${cls}`}>
      {score}
      <small>/10</small>
    </span>
  )
}

/** A number that counts up when it first appears. */
export function CountUp({ value, decimals = 0, suffix = '', duration = 800 }) {
  const [shown, setShown] = useState(0)
  useEffect(() => {
    if (value === null || value === undefined || Number.isNaN(value)) return
    let raf
    const start = performance.now()
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration)
      // ease-out so it decelerates into the final value
      setShown(value * (1 - Math.pow(1 - t, 3)))
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [value, duration])
  if (value === null || value === undefined) return <span className="mono">—</span>
  return (
    <span className="mono">
      {shown.toFixed(decimals)}
      {suffix}
    </span>
  )
}

const OUTCOME_META = [
  { key: 'home_win', cls: 'home', label: (h) => h },
  { key: 'draw', cls: 'draw', label: () => 'Draw' },
  { key: 'away_win', cls: 'away', label: (h, a) => a },
]

/**
 * Three probability bars that animate 0 -> value on mount.
 * The most likely outcome is emphasised.
 */
export function ProbabilityBars({ predictions, homeTeam, awayTeam, compact }) {
  const [ready, setReady] = useState(false)
  useEffect(() => {
    setReady(false)
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setReady(true)))
    return () => cancelAnimationFrame(id)
  }, [predictions])

  if (!predictions) return null
  const best = OUTCOME_META.reduce((a, b) =>
    predictions[a.key] >= predictions[b.key] ? a : b
  )

  return (
    <div>
      {OUTCOME_META.map((o) => {
        const pct = (predictions[o.key] || 0) * 100
        const isBest = o.key === best.key
        return (
          <div className="prob-row" key={o.key}>
            <div className={`prob-name ${isBest ? 'win' : ''}`}>
              {o.label(homeTeam || 'Home', awayTeam || 'Away')}
            </div>
            <div className="prob-track">
              <div
                className={`prob-fill ${o.cls}`}
                style={{ width: ready ? `${pct}%` : '0%', opacity: isBest ? 1 : 0.55 }}
              />
            </div>
            <div className="prob-pct" style={{ color: isBest ? 'var(--text)' : 'var(--text-dim)' }}>
              {pct.toFixed(compact ? 0 : 1)}%
            </div>
          </div>
        )
      })}
    </div>
  )
}

/** Probability cell for the match table, tinted by how likely it is. */
export function ProbCell({ value, best }) {
  const pct = (value || 0) * 100
  const cls = best ? 'high' : pct >= 30 ? 'mid' : 'low'
  return <span className={`pcell ${cls}`}>{pct.toFixed(0)}%</span>
}

export function StatTile({ label, value, sub, green }) {
  return (
    <div className="card tight">
      <div className="label">{label}</div>
      <div className={`stat-value ${green ? 'green' : ''}`} style={{ marginTop: 6 }}>
        {value}
      </div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}
