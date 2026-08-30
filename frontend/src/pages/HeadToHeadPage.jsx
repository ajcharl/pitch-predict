import { useEffect, useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { Page } from '../App'
import { EmptyState, ErrorState, SkeletonCard } from '../components/ui'
import { useH2H, useLeagues, useTeams } from '../hooks/useApi'
import { axisTick, axisTickSans, chart, tooltipStyle } from '../theme'

export default function HeadToHeadPage() {
  const { data: leaguesData } = useLeagues()
  const featured = useMemo(
    () => (leaguesData?.leagues || []),
    [leaguesData]
  )
  const [league, setLeague] = useState('E0')
  const { data: teamsData, loading: teamsLoading } = useTeams(league)
  const teams = teamsData?.teams || []

  const [t1, setT1] = useState('')
  const [t2, setT2] = useState('')
  useEffect(() => {
    if (!teams.length) return
    setT1(teams[0]?.name || '')
    setT2(teams[1]?.name || '')
  }, [teamsData])

  const { data, loading, error, retry } = useH2H(t1, t2)

  return (
    <Page title="Head to Head" subtitle="Every recorded meeting between two clubs">
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="section-head">Select teams</div>
        <div className="tabs" style={{ marginBottom: 16 }}>
          {featured.map((l) => (
            <button key={l.code} className={`tab ${league === l.code ? 'active' : ''}`}
                    onClick={() => setLeague(l.code)}>
              {l.name}
            </button>
          ))}
        </div>
        <div className="grid grid-2">
          <Picker label="Team 1" value={t1} onChange={setT1} teams={teams} exclude={t2} loading={teamsLoading} />
          <Picker label="Team 2" value={t2} onChange={setT2} teams={teams} exclude={t1} loading={teamsLoading} />
        </div>
      </div>

      {loading && <SkeletonCard rows={5} />}
      {error && <ErrorState error={error} onRetry={retry} />}
      {!loading && !error && data && (data.played ? <H2HBody data={data} /> : (
        <EmptyState
          title="No meetings on record"
          hint={`${data.team1} and ${data.team2} have not met in the data we hold (2000–2025, domestic leagues).`}
        />
      ))}
    </Page>
  )
}

function Picker({ label, value, onChange, teams, exclude, loading }) {
  return (
    <div className="field">
      <span className="label">{label}</span>
      <div className="select-wrap">
        <select value={value} onChange={(e) => onChange(e.target.value)} disabled={loading}>
          {loading && <option>Loading…</option>}
          {teams.map((t) => (
            <option key={t.name} value={t.name} disabled={t.name === exclude}>{t.name}</option>
          ))}
        </select>
      </div>
    </div>
  )
}

function H2HBody({ data }) {
  // Wins/draws/losses as a small bar chart.
  const chartData = [
    { name: shorten(data.team1), value: data.team1_wins, fill: chart.pos },
    { name: 'Draws', value: data.draws, fill: chart.warn },
    { name: shorten(data.team2), value: data.team2_wins, fill: chart.neg },
  ]

  // Goals per meeting over time, oldest first, for the trend chart.
  const timeline = [...data.meetings].reverse().map((m) => ({
    date: m.date.slice(2, 7),
    goals: m.home_goals + m.away_goals,
    label: `${m.home_team} ${m.score} ${m.away_team}`,
  }))

  const total = data.played || 1
  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <Tile label="Meetings" value={data.played} />
        <Tile label={`${data.team1} wins`} value={data.team1_wins}
              sub={`${((data.team1_wins / total) * 100).toFixed(0)}%`} green />
        <Tile label="Draws" value={data.draws} sub={`${((data.draws / total) * 100).toFixed(0)}%`} />
        <Tile label={`${data.team2} wins`} value={data.team2_wins}
              sub={`${((data.team2_wins / total) * 100).toFixed(0)}%`} />
      </div>

      <div className="grid grid-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <div className="section-head">Record</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData} margin={{ top: 6, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} vertical={false} />
              <XAxis dataKey="name" tick={axisTickSans}
                     axisLine={{ stroke: chart.axis }} tickLine={false} />
              <YAxis tick={axisTick}
                     axisLine={false} tickLine={false} allowDecimals={false} />
              <Tooltip {...tooltipStyle} cursor={{ fill: chart.cursor }} />
              <Bar dataKey="value" radius={[5, 5, 0, 0]} name="Matches">
                {chartData.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="mono" style={{ fontSize: 12, color: 'var(--text-dim)', textAlign: 'center' }}>
            Goals {data.team1_goals} – {data.team2_goals} · {data.avg_goals} per game
          </div>
        </div>

        <div className="card">
          <div className="section-head">Goals per meeting over time</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={timeline} margin={{ top: 6, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} vertical={false} />
              <XAxis dataKey="date" tick={{ ...axisTick, fontSize: 10 }}
                     axisLine={{ stroke: chart.axis }} tickLine={false} />
              <YAxis tick={axisTick}
                     axisLine={false} tickLine={false} allowDecimals={false} />
              <Tooltip {...tooltipStyle} cursor={{ fill: chart.cursor }}
                       formatter={(v, n, p) => [`${v} goals`, p.payload.label]} />
              <Bar dataKey="goals" fill={chart.accentDim} radius={[5, 5, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <div className="section-head">Recent meetings</div>
        <div className="table-wrap" style={{ border: 'none' }}>
          <table className="matches" style={{ minWidth: 560 }}>
            <thead>
              <tr>
                <th>Date</th><th>League</th>
                <th style={{ textAlign: 'right' }}>Home</th>
                <th style={{ textAlign: 'center' }}>Score</th>
                <th>Away</th><th>Winner</th>
              </tr>
            </thead>
            <tbody>
              {data.meetings.map((m, i) => (
                <tr key={i} className="fade-row" style={{ animationDelay: `${Math.min(i, 12) * 25}ms` }}>
                  <td className="mono" style={{ color: 'var(--text-dim)', fontSize: 12 }}>{m.date}</td>
                  <td style={{ color: 'var(--text-faint)', fontSize: 11.5 }}>{m.league}</td>
                  <td className="team-name" style={{ textAlign: 'right' }}>{m.home_team}</td>
                  <td style={{ textAlign: 'center' }}><span className="score-chip">{m.score}</span></td>
                  <td className="team-name">{m.away_team}</td>
                  <td>
                    <span className={`badge ${m.winner ? 'green' : 'amber'}`}>
                      {m.winner || 'Draw'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}

function Tile({ label, value, sub, green }) {
  return (
    <div className="card tight">
      <div className="label" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {label}
      </div>
      <div className={`stat-value ${green ? 'green' : ''}`} style={{ marginTop: 5 }}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}


export { tooltipStyle }

const shorten = (n) => (n.length > 12 ? n.slice(0, 11) + '…' : n)
