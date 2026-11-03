import { useState } from 'react'
import { auth } from '../api'
import MeshBackground from '../components/MeshBackground'
import ChatAssistant from '../components/ChatAssistant'

const DEMO_ACCOUNTS = [
  { email: 'doctor@radiology.ai', role: 'Doctor', desc: 'Upload & analyze scans', icon: '🩺' },
  { email: 'radiologist@radiology.ai', role: 'Radiologist', desc: 'Full access + audit logs', icon: '🔬' },
  { email: 'admin@radiology.ai', role: 'Admin', desc: 'System administration', icon: '⚙️' },
]

const FEATURES = ['DenseNet121 AI', 'Grad-CAM', '12+ Diagnoses', 'DICOM + PACS', 'PDF Reports']

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('doctor@radiology.ai')
  const [password, setPassword] = useState('demo1234')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const { data } = await auth.login(email, password)
      localStorage.setItem('token', data.access_token)
      const me = await auth.me()
      onLogin(data.access_token, me.data)
    } catch {
      setError('Invalid credentials')
    } finally {
      setLoading(false)
    }
  }

  const quickLogin = (account) => {
    setEmail(account.email)
    setPassword('demo1234')
  }

  return (
    <div className="min-h-screen relative flex items-center justify-center p-4">
      <MeshBackground />

      <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-2 gap-10 items-center relative z-10">
        <div className="space-y-6 hidden lg:block">
          <div>
            <h1 className="text-5xl font-display font-bold text-slate-900 leading-tight">
              Radiology
              <span className="block bg-gradient-to-r from-cyan-600 via-sky-600 to-indigo-600 bg-clip-text text-transparent">
                AI Assistant
              </span>
            </h1>
            <p className="text-slate-600 text-lg mt-4 max-w-md leading-relaxed">
              Explainable multi-modality clinical decision support with transparent AI insights.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            {FEATURES.map((f) => (
              <span
                key={f}
                className="px-3 py-1.5 rounded-xl text-sm text-slate-700 border border-slate-200 bg-white shadow-xs"
              >
                {f}
              </span>
            ))}
          </div>
        </div>

        <div className="card card-glow">
          <div className="mb-6">
            <h2 className="text-2xl font-display font-bold text-slate-900">Welcome back</h2>
            <p className="text-slate-500 text-sm mt-1">Sign in to your clinical workspace</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Email</label>
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="input-field" required />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Password</label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="input-field" required />
            </div>

            {error && (
              <p className="text-rose-600 text-sm bg-rose-50 border border-rose-200 rounded-xl px-4 py-2">{error}</p>
            )}

            <button type="submit" disabled={loading} className="btn-primary w-full py-3 text-base">
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Signing in...
                </span>
              ) : 'Sign In →'}
            </button>
          </form>

          <div className="mt-6 pt-5 border-t border-slate-200">
            <p className="text-xs text-slate-500 mb-3 font-medium uppercase tracking-widest">Quick Demo Login</p>
            <div className="space-y-2">
              {DEMO_ACCOUNTS.map((acc) => (
                <button
                  key={acc.email}
                  type="button"
                  onClick={() => quickLogin(acc)}
                  className="w-full text-left p-3 rounded-xl border border-slate-200 bg-slate-50 hover:bg-white hover:border-cyan-300 transition-colors group"
                >
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-sm text-slate-800 flex items-center gap-2">
                      <span className="text-base select-none">{acc.icon}</span>
                      {acc.role}
                    </span>
                    <span className="text-xs text-slate-400 group-hover:text-cyan-600 font-mono">demo1234</span>
                  </div>
                  <p className="text-xs text-slate-500 mt-0.5 ml-6">{acc.desc}</p>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <ChatAssistant />
    </div>
  )
}
