import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Page } from '../App'
import { EmptyState, ErrorState } from '../components/ui'
import { useLeagues, useTeams } from '../hooks/useApi'

/**
 * Build a hypothetical fixture between any two teams.
 *
 * The scheduled slate lives on the Fixtures page; this is for matchups that
 * are not on any schedule -- including cross-league ties, where the two clubs
 * come from different domestic leagues. Elo is calibrated across leagues, so
 * those are still a fair comparison.
 */
export default function CustomMatchPage() {
  const navigate = useNavigate()
  const { data: leaguesData } = useLeagues()
  const featured = useMemo(
    () => (leaguesData?.leagues || []),
    [leaguesData]
  )

  const [league, setLeague] = useState('E0')
  const [home, setHome] = useState('')
  const [away, setAway] = useState('')

  const { data: teamsData, loading: teamsLoading, error: teamsError, retry } = useTeams(league)
  const teams = teamsData?.teams || []

  // Preselect two strong sides so the page is usable immediately.
  useEffect(() => {
    if (!teams.length) return
    setHome(teams[0]?.name || '')
    setAway(teams[1]?.name || '')
  }, [teamsData])

  const canPredict = home && away && home !== away
  const go = () => {
    const q = new URLSearchParams({ home, away, from: 'custom' })
    navigate(`/match?${q.toString()}`)
  }

  return (
    <Page
      title="Custom matchup"
      subtitle="Predict any two teams, scheduled or not — including across leagues"
    >
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="section-head">Select fixture</div>

        <div className="tabs" style={{ marginBottom: 16 }}>
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

        {teamsError ? (
          <ErrorState error={teamsError} onRetry={retry} />
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(0,1fr) auto minmax(0,1fr) auto',
              gap: 12,
              alignItems: 'end',
            }}
            className="predict-controls"
          >
            <TeamSelect label="Home team" value={home} onChange={setHome}
                        teams={teams} exclude={away} loading={teamsLoading} />
            <div className="mono" style={{ color: 'var(--text-faint)', paddingBottom: 11, fontSize: 12 }}>
              VS
            </div>
            <TeamSelect label="Away team" value={away} onChange={setAway}
                        teams={teams} exclude={home} loading={teamsLoading} />
            <button className="btn" disabled={!canPredict} onClick={go}>
              Predict
            </button>
          </div>
        )}

        {home && away && home === away && (
          <div className="note" style={{ marginTop: 12 }}>Pick two different teams.</div>
        )}

      </div>

      <EmptyState
        icon="⚽"
        title="Looking for this week's matches?"
        hint="The Fixtures page has the full scheduled slate with a prediction on every game."
      />
    </Page>
  )
}

function TeamSelect({ label, value, onChange, teams, exclude, loading }) {
  return (
    <div className="field">
      <span className="label">{label}</span>
      <div className="select-wrap">
        <select value={value} onChange={(e) => onChange(e.target.value)} disabled={loading}>
          {loading && <option>Loading teams…</option>}
          {!loading && !teams.length && <option>No teams</option>}
          {teams.map((t) => (
            <option key={t.name} value={t.name} disabled={t.name === exclude}>
              {t.name}
              {t.elo ? `  ·  Elo ${Math.round(t.elo)}` : ''}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}
