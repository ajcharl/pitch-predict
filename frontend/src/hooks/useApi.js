// Step 8: data-fetching hooks with loading / error / retry, plus a tiny cache
// so team and league lists are fetched once and reused across pages.

import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'

const cache = new Map()

/**
 * Core hook: runs `fetcher` whenever `deps` change.
 * `key` (optional) caches the result so revisiting a page is instant.
 * `enabled` defers the call until prerequisites are ready.
 */
function useAsync(fetcher, deps, { key, enabled = true } = {}) {
  const [data, setData] = useState(() => (key && cache.has(key) ? cache.get(key) : null))
  const [loading, setLoading] = useState(enabled && !(key && cache.has(key)))
  const [error, setError] = useState(null)
  const [nonce, setNonce] = useState(0)

  const retry = useCallback(() => setNonce((n) => n + 1), [])

  useEffect(() => {
    if (!enabled) {
      setLoading(false)
      return
    }
    if (key && cache.has(key) && nonce === 0) {
      setData(cache.get(key))
      setLoading(false)
      setError(null)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    fetcher()
      .then((res) => {
        if (cancelled) return
        if (key) cache.set(key, res)
        setData(res)
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err)
          setData(null)
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, nonce])

  return { data, loading, error, retry }
}

export function useHealth() {
  return useAsync(() => api.health(), [], { key: 'health' })
}

export function useLeagues() {
  return useAsync(() => api.leagues(), [], { key: 'leagues' })
}

/** Team lists are cached per league — fetched once on first visit. */
export function useTeams(league) {
  return useAsync(() => api.teams(league), [league], {
    key: league ? `teams:${league}` : null,
    enabled: !!league,
  })
}

export function usePrediction(home, away, league, enabled) {
  return useAsync(() => api.predict(home, away, league), [home, away, league], {
    enabled: !!(enabled && home && away && home !== away),
  })
}

export function useTeamForm(team) {
  return useAsync(() => api.teamForm(team), [team], {
    key: team ? `form:${team}` : null,
    enabled: !!team,
  })
}

export function useH2H(team1, team2) {
  return useAsync(() => api.headToHead(team1, team2, 20), [team1, team2], {
    enabled: !!(team1 && team2 && team1 !== team2),
  })
}

export function useMatches(date, league) {
  return useAsync(() => api.matches(date, league), [date, league], {})
}

export function useMatchDates(league) {
  return useAsync(() => api.matchDates(league), [league], { key: `dates:${league || 'all'}` })
}

export function useFixtures(date, league) {
  return useAsync(() => api.fixtures(date, league), [date, league], {})
}

export function useFixtureDates(league) {
  return useAsync(() => api.fixtureDates(league), [league], { key: `fxdates:${league || 'all'}` })
}

export function useFreshness() {
  return useAsync(() => api.freshness(), [], { key: 'freshness' })
}

export function useModelInfo() {
  return useAsync(() => api.modelAccuracy(), [], { key: 'model' })
}
