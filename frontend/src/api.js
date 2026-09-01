// Thin fetch wrapper. Vite proxies /api to FastAPI on :8000 in dev, so we
// only ever use relative URLs and never deal with CORS during development.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// Statuses that mean "the server is still waking up", not "your request was
// wrong". A free-tier host sleeps after inactivity and answers with a gateway
// error for the first few seconds of a cold start, so these are worth retrying.
const COLD_START_STATUSES = new Set([502, 503, 504])

async function request(path, { retries = 2 } = {}) {
  let lastError
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    let res
    try {
      res = await fetch(path)
    } catch (networkError) {
      // Connection refused / DNS still resolving during a cold boot.
      lastError = new Error('Cannot reach the server.')
      lastError.coldStart = true
      if (attempt < retries) {
        await sleep(1500 * (attempt + 1))
        continue
      }
      throw lastError
    }

    let body = null
    try {
      body = await res.json()
    } catch {
      // Non-JSON response (a proxy error page, say)
    }

    if (res.ok) return body

    const err = new Error(
      (body && (body.message || body.detail)) || `Request failed (${res.status})`
    )
    err.status = res.status
    err.code = body && body.error
    err.suggestions = (body && body.suggestions) || []

    if (COLD_START_STATUSES.has(res.status) && attempt < retries) {
      err.coldStart = true
      lastError = err
      await sleep(1500 * (attempt + 1))
      continue
    }
    throw err
  }
  throw lastError
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
