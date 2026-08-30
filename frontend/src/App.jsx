import { Route, Routes } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import CustomMatchPage from './pages/CustomMatchPage'
import MatchPage from './pages/MatchPage'
import MatchdayPage from './pages/MatchdayPage'
import FixturesPage from './pages/FixturesPage'
import LeaguesPage from './pages/LeaguesPage'
import HeadToHeadPage from './pages/HeadToHeadPage'
import TeamPage from './pages/TeamPage'
import ModelInfoPage from './pages/ModelInfoPage'

export default function App() {
  return (
    <div className="app">
      <Sidebar />
      <div className="main">
        <Routes>
          {/* Upcoming fixtures are the point of the app, so they are the landing page. */}
          <Route path="/" element={<FixturesPage />} />
          <Route path="/match" element={<MatchPage />} />
          <Route path="/custom" element={<CustomMatchPage />} />
          <Route path="/predictions" element={<MatchdayPage />} />
          <Route path="/leagues" element={<LeaguesPage />} />
          <Route path="/head-to-head" element={<HeadToHeadPage />} />
          <Route path="/team/:name" element={<TeamPage />} />
          <Route path="/model" element={<ModelInfoPage />} />
          <Route path="*" element={<FixturesPage />} />
        </Routes>
      </div>
    </div>
  )
}

/** Shared page chrome: sticky top bar + content well + footer. */
export function Page({ title, subtitle, actions, live, children }) {
  return (
    <>
      <header className="topbar">
        <div>
          <h1>{title}</h1>
          {subtitle && <div className="topbar-sub">{subtitle}</div>}
        </div>
        <div className="spacer" />
        {live && <span className="live-dot">Live model</span>}
        {actions}
      </header>
      <div className="content">
        {children}
        <Footer />
      </div>
    </>
  )
}

function Footer() {
  return (
    <div className="footer">
      Data:{' '}
      <a href="https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025"
         target="_blank" rel="noreferrer">
        Club Football Match Data 2000–2025
      </a>{' '}
      (Kaggle, by Adam Gábor), built on{' '}
      <a href="https://www.football-data.co.uk/" target="_blank" rel="noreferrer">
        Football-Data.co.uk
      </a>{' '}
      results and{' '}
      <a href="http://clubelo.com/" target="_blank" rel="noreferrer">ClubElo</a> ratings.
      <div style={{ marginTop: 6 }}>
        MatchIQ predicts outcomes from historical form. Football is genuinely unpredictable —
        these are probabilities, not certainties. Not betting advice.
      </div>
    </div>
  )
}
