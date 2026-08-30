import { Link } from 'react-router-dom'
import { FormBadges } from './ui'

/** Side-by-side team form card: last 5, goals, position, Elo. */
export default function TeamCard({ stats, venue }) {
  if (!stats) return null
  const s = stats.season_stats || {}
  return (
    <div className="card">
      <div className="row" style={{ marginBottom: 12 }}>
        <span className="badge">{venue}</span>
        <Link to={`/team/${encodeURIComponent(stats.team)}`} className="team-name">
          {stats.team}
        </Link>
        <div className="spacer" />
        {stats.league_position != null && (
          <span className="badge green mono">#{stats.league_position}</span>
        )}
      </div>

      <div className="label" style={{ marginBottom: 7 }}>Last 5</div>
      <FormBadges results={stats.last_5} />

      <div className="grid grid-2" style={{ marginTop: 16, gap: 12 }}>
        <Metric label="Scored / game" value={fmt(stats.goals_scored_avg)} />
        <Metric label="Conceded / game" value={fmt(stats.goals_conceded_avg)} />
        <Metric label="Points / game" value={fmt(stats.points_per_game)} />
        <Metric label="Elo rating" value={stats.elo != null ? Math.round(stats.elo) : '—'} />
      </div>

      {s.played > 0 && (
        <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
          <div className="label" style={{ marginBottom: 6 }}>Season {stats.season}/{String((stats.season ?? 0) + 1).slice(2)}</div>
          <div className="mono" style={{ fontSize: 12.5, color: 'var(--text-dim)' }}>
            {s.played} played · <span style={{ color: 'var(--pos)' }}>{s.wins}W</span>{' '}
            <span style={{ color: 'var(--warn)' }}>{s.draws}D</span>{' '}
            <span style={{ color: 'var(--neg)' }}>{s.losses}L</span> ·{' '}
            {s.goals_scored}:{s.goals_conceded} ({s.goal_difference >= 0 ? '+' : ''}
            {s.goal_difference})
          </div>
        </div>
      )}
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className="mono" style={{ fontSize: 17, fontWeight: 700, marginTop: 3 }}>{value}</div>
    </div>
  )
}

const fmt = (v) => (v === null || v === undefined ? '—' : Number(v).toFixed(2))
