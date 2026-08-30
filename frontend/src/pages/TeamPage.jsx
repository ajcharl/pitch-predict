import { useMemo } from 'react'
import { useParams } from 'react-router-dom'
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { Page } from '../App'
import { ErrorState, FormBadges, SkeletonCard, StatTile } from '../components/ui'
import { axisTick, chart, tooltipStyle } from '../theme'
import { useTeamForm } from '../hooks/useApi'

export default function TeamPage() {
  const { name } = useParams()
  const team = decodeURIComponent(name || '')
  const { data, loading, error, retry } = useTeamForm(team)

  return (
    <Page title={team} subtitle={data ? `${data.league_name} · season ${data.season}` : 'Team profile'}>
      {loading && <SkeletonCard rows={6} />}
      {error && <ErrorState error={error} onRetry={retry} />}
      {!loading && !error && data && <Body data={data} />}
    </Page>
  )
}

function Body({ data }) {
  const s = data.season_stats || {}

  // Points-per-game trend across the recent matches we hold, oldest first.
  const trend = useMemo(() => {
    const chron = [...(data.recent_matches || [])].reverse()
    let pts = 0
    return chron.map((m, i) => {
      pts += m.result === 'W' ? 3 : m.result === 'D' ? 1 : 0
      return {
        i: i + 1,
        date: m.date.slice(5),
        ppg: +(pts / (i + 1)).toFixed(2),
        label: `${m.venue === 'H' ? 'vs' : 'at'} ${m.opponent} ${m.goals_for}-${m.goals_against}`,
      }
    })
  }, [data])

  // "Strengths": read a few feature-style numbers into plain language.
  const strengths = useMemo(() => buildStrengths(data), [data])

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <StatTile label="League position" value={data.league_position ?? '—'} green />
        <StatTile label="Elo rating" value={data.elo != null ? Math.round(data.elo) : '—'}
                  sub="Cross-league strength" />
        <StatTile label="Points / game" value={fmt(data.points_per_game)} sub="Last 5 matches" />
        <StatTile label="Clean sheets" value={data.clean_sheets_last_5} sub="In last 5" />
      </div>

      <div className="grid grid-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <div className="section-head">Season {data.season} record</div>
          <div className="grid grid-4" style={{ gap: 12 }}>
            <Mini label="Played" value={s.played} />
            <Mini label="Won" value={s.wins} color="var(--pos)" />
            <Mini label="Drawn" value={s.draws} color="var(--warn)" />
            <Mini label="Lost" value={s.losses} color="var(--neg)" />
            <Mini label="Scored" value={s.goals_scored} />
            <Mini label="Conceded" value={s.goals_conceded} />
            <Mini label="Goal diff" value={s.goal_difference >= 0 ? `+${s.goal_difference}` : s.goal_difference} />
            <Mini label="Clean sheets" value={s.clean_sheets} />
          </div>
          <div style={{ marginTop: 16 }}>
            <div className="label" style={{ marginBottom: 7 }}>Recent form (newest first)</div>
            <FormBadges results={data.last_5} />
          </div>
        </div>

        <div className="card">
          <div className="section-head">Points per game over time</div>
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={trend} margin={{ top: 8, right: 10, left: -22, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} vertical={false} />
              <XAxis dataKey="date" tick={{ ...axisTick, fontSize: 10 }}
                     axisLine={{ stroke: chart.axis }} tickLine={false} />
              <YAxis domain={[0, 3]} tick={axisTick}
                     axisLine={false} tickLine={false} />
              <Tooltip {...tooltipStyle} formatter={(v, n, p) => [`${v} ppg`, p.payload.label]} />
              <Line type="monotone" dataKey="ppg" stroke={chart.accent} strokeWidth={2.2}
                    dot={{ r: 2.5, fill: chart.accent }} activeDot={{ r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <div className="section-head">Strengths &amp; weaknesses</div>
          {strengths.map((s2, i) => (
            <div className="row" key={i} style={{ marginBottom: 10, alignItems: 'flex-start' }}>
              <span className={`badge ${s2.good ? 'green' : 'red'}`} style={{ minWidth: 62, textAlign: 'center' }}>
                {s2.good ? 'Strong' : 'Weak'}
              </span>
              <span style={{ flex: 1, fontSize: 13, color: 'var(--text-dim)' }}>{s2.text}</span>
            </div>
          ))}
          <div className="note" style={{ marginTop: 10 }}>
            Read from the same numbers the model uses as features.
          </div>
        </div>

        <div className="card">
          <div className="section-head">Recent matches</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {(data.recent_matches || []).slice(0, 12).map((m, i) => (
              <div key={i} className="row mono fade-row"
                   style={{ fontSize: 12, gap: 10, animationDelay: `${i * 25}ms` }}>
                <span style={{ color: 'var(--text-faint)', width: 76 }}>{m.date}</span>
                <span className="badge" style={{ minWidth: 26, textAlign: 'center' }}>{m.venue}</span>
                <span style={{ flex: 1, fontFamily: 'Inter', fontWeight: 500 }}>{m.opponent}</span>
                <span className="score-chip">{m.goals_for}-{m.goals_against}</span>
                <span className={`form-badge ${m.result}`} style={{ width: 21, height: 21, fontSize: 10 }}>
                  {m.result}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}

function Mini({ label, value, color }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: color || 'var(--text)', marginTop: 2 }}>
        {value ?? '—'}
      </div>
    </div>
  )
}

/** Turn the team's numbers into readable strengths, using football norms. */
function buildStrengths(d) {
  const out = []
  const gf = d.goals_scored_avg
  const ga = d.goals_conceded_avg
  const ppg = d.points_per_game
  const s = d.season_stats || {}

  if (gf != null) {
    out.push(gf >= 1.8
      ? { good: true, text: `Scoring freely — ${gf.toFixed(2)} goals per game in the last 5.` }
      : { good: gf >= 1.2, text: `Scores ${gf.toFixed(2)} per game recently${gf < 1.2 ? ', below the ~1.35 league norm' : ''}.` })
  }
  if (ga != null) {
    out.push(ga <= 0.9
      ? { good: true, text: `Miserly at the back — only ${ga.toFixed(2)} conceded per game.` }
      : { good: ga <= 1.4, text: `Concedes ${ga.toFixed(2)} per game${ga > 1.4 ? ', a defensive weak point' : ''}.` })
  }
  if (ppg != null) {
    out.push({ good: ppg >= 1.6, text: `Current form: ${ppg.toFixed(2)} points per game over the last 5.` })
  }
  if (d.clean_sheets_last_5 != null) {
    out.push({ good: d.clean_sheets_last_5 >= 2,
      text: `${d.clean_sheets_last_5} clean sheet${d.clean_sheets_last_5 === 1 ? '' : 's'} in the last 5 matches.` })
  }
  if (d.elo != null) {
    out.push({ good: d.elo >= 1700,
      text: `Elo ${Math.round(d.elo)} — ${d.elo >= 1850 ? 'elite European level' : d.elo >= 1700 ? 'strong side' : d.elo >= 1550 ? 'mid-table strength' : 'below average for a top division'}.` })
  }
  if (s.played > 0) {
    const gd = s.goal_difference
    out.push({ good: gd >= 0, text: `Season goal difference ${gd >= 0 ? '+' : ''}${gd} from ${s.played} matches.` })
  }
  return out
}

const fmt = (v) => (v === null || v === undefined ? '—' : Number(v).toFixed(2))
