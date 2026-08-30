import TeamCard from './TeamCard'
import { CountUp, ProbabilityBars, TrustBadge } from './ui'

/**
 * The full prediction view for one fixture: probabilities, verdict, both
 * teams' form, why the model called it that way, and head-to-head.
 *
 * Shared by the fixture detail page and the custom matchup builder, so a
 * scheduled match and a hypothetical one are presented identically.
 */
export default function PredictionDetail({ prediction: p, kickoff }) {
  const h2h = p.head_to_head
  return (
    <>
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="row" style={{ marginBottom: 16 }}>
          <div className="section-head" style={{ margin: 0 }}>Prediction details</div>
          <div className="spacer" />
          {kickoff && <span className="badge mono">{kickoff}</span>}
          <span className="badge">{p.league_name}</span>
          {p.cross_league && <span className="badge amber">Cross-league</span>}
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0,1.6fr) minmax(0,1fr)',
            gap: 24,
            alignItems: 'center',
          }}
          className="result-grid"
        >
          <div>
            <ProbabilityBars
              predictions={p.predictions}
              homeTeam={p.home_team}
              awayTeam={p.away_team}
            />
          </div>

          <div style={{ textAlign: 'center', borderLeft: '1px solid var(--border)', paddingLeft: 20 }}
               className="verdict">
            <div className="label">Model verdict</div>
            <div style={{ fontSize: 21, fontWeight: 700, margin: '8px 0 4px', letterSpacing: '-0.02em' }}>
              {p.predicted_outcome === 'Home Win'
                ? p.home_team
                : p.predicted_outcome === 'Away Win'
                ? p.away_team
                : 'Draw'}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 12 }}>
              {p.predicted_outcome}
            </div>
            <div className="row" style={{ justifyContent: 'center', gap: 14 }}>
              <div>
                <div className="label">Confidence</div>
                <div className="stat-value green">
                  <CountUp value={p.confidence * 100} decimals={1} suffix="%" />
                </div>
              </div>
              <div>
                <div className="label">Trust</div>
                <div style={{ marginTop: 6 }}>
                  <TrustBadge score={p.confidence_score} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginBottom: 16 }}>
        <TeamCard stats={p.home_stats} venue="HOME" />
        <TeamCard stats={p.away_stats} venue="AWAY" />
      </div>

      <div className="grid grid-2">
        <div className="card">
          <div className="section-head">Why this prediction</div>
          {p.feature_contributions?.map((f) => {
            const max = p.feature_contributions[0].importance || 1
            return (
              <div className="feat-row" key={f.feature}>
                <div>
                  <div className="feat-label">{f.label}</div>
                  <div className="feat-bar-track">
                    <div className="feat-bar" style={{ width: `${(f.importance / max) * 100}%` }} />
                  </div>
                </div>
                <div className="feat-value">{formatFeatureValue(f.value)}</div>
              </div>
            )
          })}
          <div className="note" style={{ marginTop: 12 }}>
            Bars show how much the model relies on each factor overall; numbers are this
            fixture's actual values.
          </div>
        </div>

        <div className="card">
          <div className="section-head">Head to head</div>
          {h2h?.played ? (
            <>
              <div className="row" style={{ gap: 18, marginBottom: 14 }}>
                <HeadStat label={h2h.team1} value={h2h.team1_wins} />
                <HeadStat label="Draws" value={h2h.draws} amber />
                <HeadStat label={h2h.team2} value={h2h.team2_wins} />
                <div className="spacer" />
                <HeadStat label="Avg goals" value={h2h.avg_goals} />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                {h2h.meetings.slice(0, 5).map((m, i) => (
                  <div key={i} className="row mono" style={{ fontSize: 12, gap: 10 }}>
                    <span style={{ color: 'var(--text-faint)', width: 74 }}>{m.date}</span>
                    <span style={{ flex: 1, textAlign: 'right' }}>{m.home_team}</span>
                    <span className="score-chip">{m.score}</span>
                    <span style={{ flex: 1 }}>{m.away_team}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>
              {h2h?.note || 'No recent meetings on record.'}
              {p.cross_league && (
                <div className="note" style={{ marginTop: 12 }}>
                  These teams play in different leagues, so they have no shared history. The
                  prediction leans on Elo, which is calibrated across leagues.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  )
}

/** Feature values span Elo (~1900) to rates (0-1), so scale the precision. */
function formatFeatureValue(v) {
  if (v === null || v === undefined) return '—'
  const n = Number(v)
  if (Number.isNaN(n)) return '—'
  if (Math.abs(n) >= 100) return Math.round(n).toLocaleString()
  if (Number.isInteger(n)) return String(n)
  return n.toFixed(2)
}

function HeadStat({ label, value, amber }) {
  return (
    <div>
      <div className="label" style={{ maxWidth: 110, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {label}
      </div>
      <div className="mono" style={{ fontSize: 20, fontWeight: 700, color: amber ? 'var(--warn)' : 'var(--text)' }}>
        {value}
      </div>
    </div>
  )
}
