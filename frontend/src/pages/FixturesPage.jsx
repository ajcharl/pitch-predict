import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Page } from '../App'
import {
  EmptyState, ErrorState, FormBadges, ProbCell, SkeletonTable, TrustBadge, WakingNotice,
} from '../components/ui'
import { useFixtureDates, useFixtures, useLeagues } from '../hooks/useApi'

/**
 * Upcoming fixtures with the model's call on each. Unlike the matchday page
 * these have not been played, so there is no result column — just the
 * prediction and how much the model trusts it.
 */
export default function FixturesPage() {
  const navigate = useNavigate()
  const { data: leaguesData } = useLeagues()
  const featured = useMemo(
    () => (leaguesData?.leagues || []),
    [leaguesData]
  )

  const [league, setLeague] = useState('')
  const [date, setDate] = useState(null)
  const { data: datesData } = useFixtureDates(league)

  // Default to "all dates" so the page opens showing the whole upcoming slate.
  useEffect(() => { setDate(null) }, [league])

  const { data, loading, error, retry, slow } = useFixtures(date, league)
  const fixtures = data?.fixtures || []
  const fresh = data?.data_freshness
  const dates = datesData?.dates || []

  // Clicking a row opens the full prediction detail for that fixture.
  const openFixture = (f) => {
    if (!f.predictable) return
    const q = new URLSearchParams({ home: f.home_team, away: f.away_team, date: f.date })
    if (f.time) q.set('time', f.time)
    navigate(`/match?${q.toString()}`)
  }

  return (
    <Page
      title="Fixtures"
      subtitle="The model's call on matches that have not been played yet"
      live
      actions={
        data?.summary && (
          <span className="badge mono">
            {data.summary.predictable}/{data.summary.total} predictable
          </span>
        )
      }
    >
      {fresh && <FreshnessBar fresh={fresh} />}

      <div className="card tight" style={{ marginBottom: 16 }}>
        <div className="row" style={{ marginBottom: 12 }}>
          <div className="tabs">
            <button className={`tab ${league === '' ? 'active' : ''}`} onClick={() => setLeague('')}>
              All leagues
            </button>
            {featured.map((l) => (
              <button key={l.code} className={`tab ${league === l.code ? 'active' : ''}`}
                      onClick={() => setLeague(l.code)}>
                {l.name}
              </button>
            ))}
          </div>
        </div>
        <div className="pills">
          <button className={`pill ${date === null ? 'active' : ''}`} onClick={() => setDate(null)}>
            All upcoming
          </button>
          {dates.map((d) => (
            <button key={d.date} className={`pill ${d.date === date ? 'active' : ''}`}
                    onClick={() => setDate(d.date)} title={`${d.matches} fixtures`}>
              {formatPill(d.date)}
            </button>
          ))}
        </div>
      </div>

      {loading && slow && <WakingNotice />}
      {loading && <SkeletonTable rows={10} />}
      {error && <ErrorState error={error} onRetry={retry} />}
      {!loading && !error && !fixtures.length && (
        <EmptyState
          title="No fixtures loaded"
          hint={data?.note || 'Run `python model/refresh.py` to download the coming week.'}
        />
      )}
      {!loading && !error && fixtures.length > 0 && (
        <FixtureTable fixtures={fixtures} onOpen={openFixture} />
      )}
    </Page>
  )
}

function FreshnessBar({ fresh }) {
  const stale = !fresh.live_state
  return (
    <div className="note" style={{ marginBottom: 16 }}>
      {stale ? (
        <>
          <strong style={{ color: 'var(--warn)' }}>Using the training snapshot.</strong>{' '}
          Team form is as of {fresh.last_result}. Run <code>python model/refresh.py</code> to pull
          recent results, current Elo and this week's fixtures.
        </>
      ) : (
        <>
          Form and standings include results up to{' '}
          <strong style={{ color: 'var(--text)' }}>{fresh.last_result}</strong>; Elo refreshed{' '}
          <strong style={{ color: 'var(--text)' }}>{fresh.elo_refreshed_at}</strong>.
          {fresh.teams_with_stale_elo > 0 && (
            <> {fresh.teams_with_stale_elo} smaller-league clubs aren't in ClubElo's daily
            snapshot and keep their last known rating — those rows are marked.</>
          )}
        </>
      )}
    </div>
  )
}

function FixtureTable({ fixtures, onOpen }) {
  return (
    <div className="table-wrap">
      <table className="matches">
        <thead>
          <tr>
            <th>Kick-off</th>
            <th>League</th>
            <th>Fixture</th>
            <th style={{ textAlign: 'center' }}>Form</th>
            <th style={{ textAlign: 'center' }}>1</th>
            <th style={{ textAlign: 'center' }}>X</th>
            <th style={{ textAlign: 'center' }}>2</th>
            <th>Model pick</th>
            <th style={{ textAlign: 'center' }}>Trust</th>
          </tr>
        </thead>
        <tbody>
          {fixtures.map((f, i) => {
            if (!f.predictable) {
              return (
                <tr key={i} className="fade-row">
                  <td className="mono" style={{ color: 'var(--text-faint)', fontSize: 11.5 }}>
                    {shortDate(f.date)} {f.time || ''}
                  </td>
                  <td style={{ color: 'var(--text-faint)', fontSize: 11.5 }}>{f.league_name}</td>
                  <td>
                    <div className="fixture">
                      <span className="team-name" style={{ textAlign: 'right', flex: 1 }}>{f.home_team}</span>
                      <span className="vs">v</span>
                      <span className="team-name" style={{ flex: 1 }}>{f.away_team}</span>
                    </div>
                  </td>
                  <td colSpan={6} style={{ color: 'var(--text-faint)', fontSize: 12 }}>
                    <span className="badge">No prediction</span>{' '}
                    <span title={f.reason}>not enough history for this club</span>
                  </td>
                </tr>
              )
            }
            const p = f.predictions
            const best = f.predicted_code
            return (
              <tr
                key={i}
                className="fade-row clickable"
                style={{ animationDelay: `${Math.min(i, 14) * 22}ms` }}
                onClick={() => onOpen(f)}
                title="Open full prediction"
              >
                <td className="mono" style={{ color: 'var(--text-dim)', fontSize: 11.5, whiteSpace: 'nowrap' }}>
                  {shortDate(f.date)}<br />
                  <span style={{ color: 'var(--text-faint)' }}>{f.time || ''}</span>
                </td>
                <td style={{ color: 'var(--text-faint)', fontSize: 11.5, whiteSpace: 'nowrap' }}>
                  {f.league_name}
                  {f.elo_stale && (
                    <div title="One club is not in ClubElo's daily snapshot; using its last known rating">
                      <span className="badge amber" style={{ fontSize: 9 }}>stale elo</span>
                    </div>
                  )}
                </td>
                <td>
                  <div className="fixture">
                    <Link to={`/team/${encodeURIComponent(f.home_team)}`} className="team-name"
                          onClick={(e) => e.stopPropagation()}
                          style={{ textAlign: 'right', flex: 1 }}>{f.home_team}</Link>
                    <span className="vs">v</span>
                    <Link to={`/team/${encodeURIComponent(f.away_team)}`} className="team-name"
                          onClick={(e) => e.stopPropagation()}
                          style={{ flex: 1 }}>{f.away_team}</Link>
                  </div>
                  <div className="mono" style={{ fontSize: 10, color: 'var(--text-faint)', textAlign: 'center', marginTop: 2 }}>
                    {Math.round(f.home_elo)} · {Math.round(f.away_elo)}
                  </div>
                </td>
                <td>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 3, alignItems: 'center' }}>
                    <FormBadges results={(f.home_last_5 || []).slice(0, 5)} size={17} />
                    <FormBadges results={(f.away_last_5 || []).slice(0, 5)} size={17} />
                  </div>
                </td>
                <td style={{ textAlign: 'center' }}><ProbCell value={p.home_win} best={best === 'H'} /></td>
                <td style={{ textAlign: 'center' }}><ProbCell value={p.draw} best={best === 'D'} /></td>
                <td style={{ textAlign: 'center' }}><ProbCell value={p.away_win} best={best === 'A'} /></td>
                <td><span className="badge green">{f.predicted_outcome}</span></td>
                <td style={{ textAlign: 'center' }}><TrustBadge score={f.confidence_score} /></td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function shortDate(iso) {
  const d = new Date(iso + 'T00:00:00')
  return d.toLocaleDateString('en-GB', { weekday: 'short', day: '2-digit', month: 'short' })
}

function formatPill(iso) {
  const d = new Date(iso + 'T00:00:00')
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const diff = Math.round((d - today) / 86400000)
  if (diff === 0) return 'Today'
  if (diff === 1) return 'Tomorrow'
  if (diff === -1) return 'Yesterday'
  return d.toLocaleDateString('en-GB', { weekday: 'short', day: '2-digit', month: 'short' })
}
