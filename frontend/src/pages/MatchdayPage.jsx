import { useEffect, useMemo, useState } from 'react'
import { Page } from '../App'
import {
  EmptyState, ErrorState, ProbCell, SkeletonTable, TrustBadge,
} from '../components/ui'
import { useLeagues, useMatchDates, useMatches } from '../hooks/useApi'

/**
 * Browse real matchdays with the model's out-of-sample call next to the actual
 * result. Every match here is from the 2023-2025 test period, so nothing shown
 * was part of training.
 */
export default function MatchdayPage() {
  const { data: leaguesData } = useLeagues()
  const featured = useMemo(
    () => (leaguesData?.leagues || []),
    [leaguesData]
  )

  const [league, setLeague] = useState('')
  const { data: datesData } = useMatchDates(league)
  const [date, setDate] = useState(null)

  // When the league changes the available dates change too — snap to the most
  // recent one so the table is never empty.
  useEffect(() => {
    if (datesData?.latest) setDate(datesData.latest)
  }, [datesData])

  const { data, loading, error, retry } = useMatches(date, league)
  const matches = data?.matches || []
  const summary = data?.summary

  // Show a window of dates around the selected one as pills.
  const pillDates = useMemo(() => {
    const all = datesData?.dates || []
    if (!date) return all.slice(0, 9)
    const idx = all.findIndex((d) => d.date === date)
    const start = Math.max(0, Math.min(idx - 4, all.length - 9))
    return all.slice(start, start + 9)
  }, [datesData, date])

  const allDates = datesData?.dates || []
  const idx = allDates.findIndex((d) => d.date === date)
  const step = (delta) => {
    const next = allDates[idx + delta]
    if (next) setDate(next.date)
  }

  return (
    <Page
      title="Predictions by matchday"
      subtitle="Model calls vs actual results, on matches it never trained on"
      actions={
        summary?.accuracy != null && (
          <span className="badge green mono">
            {summary.correct}/{summary.played} correct · {(summary.accuracy * 100).toFixed(0)}%
          </span>
        )
      }
    >
      {/* Date navigation */}
      <div className="card tight" style={{ marginBottom: 16 }}>
        <div className="row" style={{ marginBottom: 12 }}>
          <div className="tabs">
            <button className={`tab ${league === '' ? 'active' : ''}`} onClick={() => setLeague('')}>
              All leagues
            </button>
            {featured.map((l) => (
              <button
                key={l.code}
                className={`tab ${league === l.code ? 'active' : ''}`}
                onClick={() => setLeague(l.code)}
              >
                {l.name}
              </button>
            ))}
          </div>
        </div>

        <div className="row" style={{ gap: 8 }}>
          <button className="btn ghost sm" onClick={() => step(1)} disabled={idx < 0 || idx >= allDates.length - 1}>
            ‹ Earlier
          </button>
          <div className="pills" style={{ flex: 1 }}>
            {pillDates.map((d) => (
              <button
                key={d.date}
                className={`pill ${d.date === date ? 'active' : ''}`}
                onClick={() => setDate(d.date)}
                title={`${d.matches} matches`}
              >
                {formatPill(d.date)}
              </button>
            ))}
          </div>
          <button className="btn ghost sm" onClick={() => step(-1)} disabled={idx <= 0}>
            Later ›
          </button>
        </div>
      </div>

      {loading && <SkeletonTable rows={9} />}
      {error && <ErrorState error={error} onRetry={retry} />}
      {!loading && !error && !matches.length && (
        <EmptyState title="No matches on this date" hint="Pick another date above." />
      )}
      {!loading && !error && matches.length > 0 && <MatchTable matches={matches} />}
    </Page>
  )
}

export function MatchTable({ matches }) {
  return (
    <div className="table-wrap">
      <table className="matches">
        <thead>
          <tr>
            <th>League</th>
            <th>Fixture</th>
            <th style={{ textAlign: 'center' }}>1</th>
            <th style={{ textAlign: 'center' }}>X</th>
            <th style={{ textAlign: 'center' }}>2</th>
            <th>Model pick</th>
            <th style={{ textAlign: 'center' }}>Trust</th>
            <th style={{ textAlign: 'center' }}>Result</th>
          </tr>
        </thead>
        <tbody>
          {matches.map((m, i) => {
            const p = m.predictions
            const best = m.predicted_code
            return (
              <tr key={i} className="fade-row" style={{ animationDelay: `${Math.min(i, 14) * 22}ms` }}>
                <td style={{ color: 'var(--text-faint)', fontSize: 11.5, whiteSpace: 'nowrap' }}>
                  {m.league_name}
                </td>
                <td>
                  <div className="fixture">
                    <span className="team-name" style={{ textAlign: 'right', flex: 1 }}>{m.home_team}</span>
                    <span className="vs">v</span>
                    <span className="team-name" style={{ flex: 1 }}>{m.away_team}</span>
                  </div>
                </td>
                <td style={{ textAlign: 'center' }}><ProbCell value={p.home_win} best={best === 'H'} /></td>
                <td style={{ textAlign: 'center' }}><ProbCell value={p.draw} best={best === 'D'} /></td>
                <td style={{ textAlign: 'center' }}><ProbCell value={p.away_win} best={best === 'A'} /></td>
                <td>
                  <span className="badge green">{m.predicted_outcome}</span>
                </td>
                <td style={{ textAlign: 'center' }}><TrustBadge score={m.confidence_score} /></td>
                <td style={{ textAlign: 'center' }}>
                  <span className={`badge ${m.correct ? 'green' : 'red'}`}>
                    {m.actual_outcome}
                  </span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function formatPill(iso) {
  const d = new Date(iso + 'T00:00:00')
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
}
