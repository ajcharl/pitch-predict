import { Link, useSearchParams } from 'react-router-dom'
import { Page } from '../App'
import PredictionDetail from '../components/PredictionDetail'
import { ErrorState, Loading } from '../components/ui'
import { usePrediction } from '../hooks/useApi'

/**
 * Detail view for a single fixture, addressed by query string so team names
 * with spaces or apostrophes ("Nott'm Forest") cannot break the route.
 *
 *   /match?home=Liverpool&away=Nottm+Forest&date=2026-08-29&time=12:30
 */
export default function MatchPage() {
  const [params] = useSearchParams()
  const home = params.get('home') || ''
  const away = params.get('away') || ''
  const date = params.get('date')
  const time = params.get('time')
  const back = params.get('from') === 'custom' ? '/custom' : '/'

  const { data, loading, error, retry } = usePrediction(home, away, null, true)

  const kickoff = date
    ? `${new Date(date + 'T00:00:00').toLocaleDateString('en-GB', {
        weekday: 'short', day: '2-digit', month: 'short',
      })}${time ? ` · ${time}` : ''}`
    : null

  return (
    <Page
      title={home && away ? `${home} v ${away}` : 'Match'}
      subtitle={kickoff ? `Kick-off ${kickoff}` : 'Prediction detail'}
      actions={
        <Link to={back} className="btn ghost sm">
          {back === '/custom' ? '← Back to builder' : '← All fixtures'}
        </Link>
      }
    >
      {(!home || !away) && (
        <ErrorState error={{ message: 'No fixture specified.' }} />
      )}
      {home && away && loading && <Loading />}
      {home && away && error && <ErrorState error={error} onRetry={retry} />}
      {home && away && !loading && !error && data && (
        <PredictionDetail prediction={data} kickoff={kickoff} />
      )}
    </Page>
  )
}
