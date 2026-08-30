// Thin fetch wrapper. Vite proxies /api to FastAPI on :8000 in dev, so we
// only ever use relative URLs and never deal with CORS during development.

async function request(path) {
  const res = await fetch(path)
  let body = null
  try {
    body = await res.json()
  } catch {
    // Non-JSON response (a proxy error page, say)
  }
  if (!res.ok) {
    const err = new Error(
      (body && (body.message || body.detail)) || `Request failed (${res.status})`
    )
    err.status = res.status
    err.code = body && body.error
    err.suggestions = (body && body.suggestions) || []
    throw err
  }
  return body
}

const qs = (params) => {
  const sp = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== '') sp.set(k, v)
  })
  const s = sp.toString()
  return s ? `?${s}` : ''
}

export const api = {
  health: () => request('/api/health'),
  leagues: () => request('/api/leagues'),
  teams: (league) => request(`/api/teams${qs({ league })}`),
  predict: (home, away, league) => request(`/api/predict${qs({ home, away, league })}`),
  teamForm: (team) => request(`/api/team/${encodeURIComponent(team)}/form`),
  headToHead: (team1, team2, limit) => request(`/api/head-to-head${qs({ team1, team2, limit })}`),
  matches: (date, league) => request(`/api/matches${qs({ date, league })}`),
  matchDates: (league) => request(`/api/matches/dates${qs({ league })}`),
  fixtures: (date, league) => request(`/api/fixtures${qs({ date, league })}`),
  fixtureDates: (league) => request(`/api/fixtures/dates${qs({ league })}`),
  freshness: () => request('/api/freshness'),
  modelAccuracy: () => request('/api/model/accuracy'),
}
