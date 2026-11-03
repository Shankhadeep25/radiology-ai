import { NavLink, Outlet } from 'react-router-dom'
import MeshBackground from './MeshBackground'
import ChatAssistant from './ChatAssistant'
import { useStudyChat } from '../context/StudyChatContext'

export default function Layout({ user, onLogout }) {
  const { studyContext } = useStudyChat()

  const navClass = ({ isActive }) =>
    `px-4 py-1.5 rounded-xl text-sm font-medium transition-colors ${
      isActive
        ? 'bg-cyan-50 text-cyan-800 border border-cyan-200 font-semibold shadow-xs'
        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100 border border-transparent'
    }`

  return (
    <div className="min-h-screen relative font-sans">
      <MeshBackground />

      <header className="glass-nav sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 gap-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-600 to-indigo-600 flex items-center justify-center shadow-xs">
                <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-base sm:text-lg font-bold tracking-tight text-slate-900 leading-tight">Radiology AI</h1>
                <p className="text-[10px] uppercase tracking-widest text-cyan-700 font-semibold leading-tight">Clinical CDS</p>
              </div>
            </div>

            <nav className="hidden md:flex items-center gap-1.5">
              <NavLink to="/" className={navClass} end>Dashboard</NavLink>
              <NavLink to="/analyze" className={navClass}>Analyze</NavLink>
              <NavLink to="/patients" className={navClass}>Patients</NavLink>
              <NavLink to="/history" className={navClass}>History</NavLink>
              {(user.role === 'admin' || user.role === 'radiologist') && (
                <NavLink to="/audit" className={navClass}>Audit</NavLink>
              )}
            </nav>

            <div className="flex items-center gap-3">
              <div className="text-right hidden sm:block">
                <p className="text-sm font-semibold text-slate-800 leading-tight">{user.full_name}</p>
                <p className="text-[11px] text-slate-500 capitalize leading-tight mt-0.5">{user.role}</p>
              </div>
              <button onClick={onLogout} className="btn-secondary text-sm font-medium px-3.5 py-1.5 rounded-lg">Logout</button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>

      <ChatAssistant studyContext={studyContext} />
    </div>
  )
}
