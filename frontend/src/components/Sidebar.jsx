import { NavLink } from 'react-router-dom'

// Inline SVG icons — no icon library, so nothing extra to load.
const Icon = ({ d, circle }) => (
  <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    {circle && <circle cx="12" cy="12" r="9" />}
    <path d={d} />
  </svg>
)

const NAV = [
  { to: '/', end: true, label: 'Fixtures', icon: <Icon d="M4 6h16M4 6v13h16V6M8 3v4M16 3v4M9 12h2M9 16h2M14 12h2" /> },
  { to: '/custom', label: 'Custom matchup', icon: <Icon d="M12 3v18M7 7l-3 5 3 5M17 7l3 5-3 5" /> },
  { to: '/predictions', label: 'Track record', icon: <Icon d="M3 17l5.5-6 4 4L21 6M21 6h-5m5 0v5" /> },
  { to: '/leagues', label: 'Leagues', icon: <Icon d="M4 5h16M4 5v6a8 8 0 0 0 16 0V5M9 21h6M12 19v2" /> },
  { to: '/head-to-head', label: 'Head to Head', icon: <Icon d="M9 3v18M15 3v18M3 9h18M3 15h18" /> },
  { to: '/model', label: 'Model Info', icon: <Icon d="M4 20V10M10 20V4M16 20v-7M22 20H2" /> },
]

export default function Sidebar() {
  return (
    <nav className="sidebar">
      <div className="brand">
        <div className="brand-mark">⚽</div>
        <div>
          <div className="brand-name">MatchIQ</div>
          <div className="brand-sub">Match Intelligence</div>
        </div>
      </div>

      <div className="nav-label">Navigate</div>
      {NAV.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.end}
          className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
        >
          {item.icon}
          <span>{item.label}</span>
        </NavLink>
      ))}

      <div className="sidebar-foot">
        <div className="mono" style={{ fontSize: 10, letterSpacing: '0.1em' }}>
          RANDOM FOREST · 130K MATCHES
        </div>
        <div style={{ marginTop: 4 }}>Trained on 2000–2022, tested on 2023–2025.</div>
      </div>
    </nav>
  )
}
