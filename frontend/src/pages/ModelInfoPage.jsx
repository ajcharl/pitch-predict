import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { Page } from '../App'
import { CountUp, ErrorState, SkeletonCard, StatTile } from '../components/ui'
import { axisTick, axisTickSans, chart, tooltipStyle } from '../theme'
import { useModelInfo } from '../hooks/useApi'

export default function ModelInfoPage() {
  const { data, loading, error, retry } = useModelInfo()
  return (
    <Page title="Model Info" subtitle="How MatchIQ predicts, and how well it does">
      {loading && <SkeletonCard rows={7} />}
      {error && <ErrorState error={error} onRetry={retry} />}
      {!loading && !error && data && <Body m={data} />}
    </Page>
  )
}

function Body({ m }) {
  const acc = (m.accuracy || 0) * 100
  const base = (m.baseline_accuracy || 0) * 100
  const labels = m.labels || ['H', 'D', 'A']
  const names = m.label_names || {}

  const leagueData = Object.entries(m.per_league || {})
    .map(([code, v]) => ({
      code, name: v.name,
      accuracy: +(v.accuracy * 100).toFixed(1),
      baseline: +(v.baseline * 100).toFixed(1),
      matches: v.matches,
    }))
    .sort((a, b) => b.accuracy - a.accuracy)

  const featData = (m.feature_importance || []).slice(0, 14).map((f) => ({
    name: f.feature,
    label: (m.feature_descriptions || {})[f.feature] || f.feature,
    importance: +(f.importance * 100).toFixed(2),
  }))

  return (
    <>
      {/* Headline numbers */}
      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <StatTile label="Accuracy" green
                  value={<CountUp value={acc} decimals={1} suffix="%" />}
                  sub={`${m.test_matches?.toLocaleString()} unseen matches`} />
        <StatTile label="Baseline" value={<CountUp value={base} decimals={1} suffix="%" />}
                  sub="Always predict home win" />
        <StatTile label="Edge over baseline"
                  value={<CountUp value={acc - base} decimals={1} suffix=" pts" />}
                  sub="Percentage points" green />
        <StatTile label="Log loss" value={m.log_loss?.toFixed(3)}
                  sub={`vs ${m.uniform_log_loss} for a blind guess`} />
      </div>

      {/* Plain English */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-head">How it works, in plain English</div>
        <div style={{ fontSize: 13.5, color: 'var(--text-dim)', lineHeight: 1.75, maxWidth: 900 }}>
          <p style={{ marginTop: 0 }}>
            MatchIQ never looks at a match it is predicting. For every fixture it describes the two
            teams <strong style={{ color: 'var(--text)' }}>as they were beforehand</strong>: how
            many points they took in their last five games, how many goals they scored and
            conceded, how they do specifically at home or away, their league position and goal
            difference, their record against this exact opponent, and their{' '}
            <strong style={{ color: 'var(--text)' }}>Elo rating</strong> — a single number for team
            strength that goes up when you win and down when you lose, and that moves more when you
            beat someone good.
          </p>
          <p>
            Those {Object.keys(m.feature_descriptions || {}).length} numbers go to a{' '}
            <strong style={{ color: 'var(--text)' }}>Random Forest</strong>: 400 decision trees,
            each shown a random slice of the history, that vote on the outcome. The share of the
            vote each result gets becomes the probability you see.
          </p>
          <p>
            It learned on seasons {m.train_seasons?.[0]}–{m.train_seasons?.[1]} (
            {m.train_matches?.toLocaleString()} matches from Europe's big five) and is scored on
            seasons {m.test_seasons?.[0]}–{m.test_seasons?.[1]} (
            {m.test_matches?.toLocaleString()} matches) that came strictly later — the same test
            it faces in real life.
          </p>
          <p style={{ marginBottom: 0 }}>
            <strong style={{ color: 'var(--text)' }}>Is {acc.toFixed(1)}% good?</strong> Yes. A
            three-way guess is 33%. Always backing the home team gets {base.toFixed(1)}%. Serious
            commercial models reach roughly 52–56%, because football is genuinely, irreducibly
            unpredictable — a deflection decides a match no model can foresee.
          </p>
        </div>
      </div>

      {/* Per-class + confusion matrix */}
      <div className="grid grid-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <div className="section-head">Accuracy by outcome</div>
          {labels.map((c) => {
            const pc = (m.per_class || {})[c]
            if (!pc) return null
            return (
              <div key={c} style={{ marginBottom: 14 }}>
                <div className="row" style={{ marginBottom: 5 }}>
                  <span style={{ fontSize: 13, fontWeight: 500 }}>{pc.name}</span>
                  <div className="spacer" />
                  <span className="mono" style={{ fontSize: 12.5, color: 'var(--text-dim)' }}>
                    recall {(pc.recall * 100).toFixed(1)}% · precision {(pc.precision * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="prob-track">
                  <div className="prob-fill" style={{
                    width: `${pc.recall * 100}%`,
                    background: c === 'H'
                      ? 'linear-gradient(90deg,var(--pos-deep),var(--pos))'
                      : c === 'D'
                      ? 'linear-gradient(90deg,var(--warn-deep),var(--warn))'
                      : 'linear-gradient(90deg,var(--neg-deep),var(--neg))',
                  }} />
                </div>
                <div style={{ fontSize: 11.5, color: 'var(--text-faint)', marginTop: 4 }}>
                  {pc.support.toLocaleString()} matches in the test set
                </div>
              </div>
            )
          })}
          <div className="note">
            <strong style={{ color: 'var(--text)' }}>Why draws are almost never the top pick:</strong>{' '}
            a draw is rarely the single most likely result — it usually sits near 25–28% while one
            side leads. So the model's headline pick is nearly always a win, even though its draw
            probability is meaningful. Read the three bars, not just the verdict.
          </div>
        </div>

        <div className="card">
          <div className="section-head">Confusion matrix</div>
          <div style={{ overflowX: 'auto' }}>
            <table className="cm">
              <thead>
                <tr>
                  <th style={{ border: 'none' }} />
                  <th colSpan={labels.length}>Predicted</th>
                </tr>
                <tr>
                  <th>Actual</th>
                  {labels.map((c) => <th key={c}>{names[c]}</th>)}
                </tr>
              </thead>
              <tbody>
                {(m.confusion_matrix || []).map((row, i) => (
                  <tr key={i}>
                    <th>{names[labels[i]]}</th>
                    {row.map((v, j) => (
                      <td key={j} className={i === j ? 'diag' : ''}>{v.toLocaleString()}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="note" style={{ marginTop: 12 }}>
            Rows are what actually happened, columns what the model said. The green diagonal is
            where it was right.
          </div>
        </div>
      </div>

      {/* Feature importance */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-head">What the model leans on (feature importance)</div>
        <ResponsiveContainer width="100%" height={Math.max(300, featData.length * 26)}>
          <BarChart data={featData} layout="vertical" margin={{ top: 4, right: 26, left: 108, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} horizontal={false} />
            <XAxis type="number" tick={axisTick}
                   axisLine={false} tickLine={false} unit="%" />
            <YAxis type="category" dataKey="name" width={104}
                   tick={{ ...axisTick, fontSize: 10.5 }}
                   axisLine={false} tickLine={false} />
            <Tooltip {...tooltipStyle} cursor={{ fill: chart.cursor }}
                     formatter={(v, n, p) => [`${v}%`, p.payload.label]} />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
              {featData.map((e, i) => (
                <Cell key={i} fill={i === 0 ? chart.accent : i < 4 ? chart.accentDim : chart.accentDeep} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="note">
          Elo dominates because it already compresses years of results into one number. Season goal
          difference and points per game add current-campaign context; form and head-to-head refine
          the margins.
        </div>
      </div>

      {/* Per league */}
      <div className="card">
        <div className="section-head">Accuracy by league (test period)</div>
        <ResponsiveContainer width="100%" height={Math.max(300, leagueData.length * 28)}>
          <BarChart data={leagueData} layout="vertical" margin={{ top: 4, right: 26, left: 128, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} horizontal={false} />
            <XAxis type="number" domain={[0, 65]} unit="%"
                   tick={axisTick}
                   axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" width={124}
                   tick={axisTickSans}
                   axisLine={false} tickLine={false} />
            <Tooltip {...tooltipStyle} cursor={{ fill: chart.cursor }}
                     formatter={(v, n) => [`${v}%`, n === 'accuracy' ? 'Model' : 'Always home win']} />
            <Bar dataKey="baseline" fill={chart.muted} radius={[0, 3, 3, 0]} name="baseline" />
            <Bar dataKey="accuracy" radius={[0, 4, 4, 0]} name="accuracy">
              {leagueData.map((e, i) => (
                <Cell key={i} fill={e.accuracy - e.baseline > 8 ? chart.accent : e.accuracy > e.baseline ? chart.accentDim : chart.warn} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="note">
          The model does best in the big European leagues, where Elo ratings are most reliable and
          squads are most stable. Second tiers are harder: more turnover, more randomness.
        </div>
      </div>
    </>
  )
}
