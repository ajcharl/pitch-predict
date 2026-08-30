import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Page } from '../App'
import { ErrorState, SkeletonTable } from '../components/ui'
import { useLeagues, useTeams } from '../hooks/useApi'

/** Browse every league the model serves, and jump into any team's profile. */
export default function LeaguesPage() {
  const { data: leaguesData, loading: lLoading } = useLeagues()
  const leagues = leaguesData?.leagues || []
  const [league, setLeague] = useState('E0')
  const { data, loading, error, retry } = useTeams(league)
  const teams = data?.teams || []

  return (
    <Page
      title="Leagues"
      subtitle={`${leagues.length} leagues · ${teams.length} teams in view`}
    >
      <div className="card tight" style={{ marginBottom: 16 }}>
        <div className="section-head">Leagues</div>
        <div className="pills">
          {lLoading && <span style={{ color: 'var(--text-faint)', fontSize: 12 }}>Loading…</span>}
          {leagues.map((l) => (
            <button
              key={l.code}
              className={`pill ${league === l.code ? 'active' : ''}`}
              onClick={() => setLeague(l.code)}
              title={`${l.teams} teams`}
            >
              {l.name}
            </button>
          ))}
        </div>
      </div>


      {loading && <SkeletonTable rows={10} />}
      {error && <ErrorState error={error} onRetry={retry} />}
      {!loading && !error && (
        <div className="table-wrap">
          <table className="matches" style={{ minWidth: 620 }}>
            <thead>
              <tr>
                <th style={{ width: 54 }}>#</th>
                <th>Team</th>
                <th>League</th>
                <th style={{ textAlign: 'right' }}>Elo</th>
                <th style={{ textAlign: 'right' }}>Matches on record</th>
              </tr>
            </thead>
            <tbody>
              {teams.map((t, i) => (
                <tr key={t.name} className="fade-row clickable"
                    style={{ animationDelay: `${Math.min(i, 16) * 18}ms` }}>
                  <td className="mono" style={{ color: 'var(--text-faint)' }}>{i + 1}</td>
                  <td>
                    <Link to={`/team/${encodeURIComponent(t.name)}`} className="team-name">
                      {t.name}
                    </Link>
                  </td>
                  <td style={{ color: 'var(--text-faint)', fontSize: 11.5 }}>{t.league_name}</td>
                  <td className="mono" style={{ textAlign: 'right', fontWeight: 600 }}>
                    {t.elo != null ? Math.round(t.elo) : '—'}
                  </td>
                  <td className="mono" style={{ textAlign: 'right', color: 'var(--text-dim)' }}>
                    {t.matches_played.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Page>
  )
}
