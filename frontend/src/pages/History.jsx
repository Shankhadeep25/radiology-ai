import { useEffect, useState } from 'react'
import { dashboard } from '../api'

export default function History() {
  const [history, setHistory] = useState([])

  useEffect(() => {
    dashboard.history().then((r) => setHistory(r.data))
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <h2 className="page-title">Study History</h2>
        <p className="page-subtitle">All analyzed chest X-ray studies</p>
      </div>

      <div className="card overflow-x-auto">
        <table className="w-full text-sm table-glass">
          <thead>
            <tr>
              <th className="pb-3 pr-4 text-left">#</th>
              <th className="pb-3 pr-4 text-left">Patient</th>
              <th className="pb-3 pr-4 text-left">Finding</th>
              <th className="pb-3 pr-4 text-left">Confidence</th>
              <th className="pb-3 text-left">Date</th>
            </tr>
          </thead>
          <tbody>
            {history.length === 0 ? (
              <tr><td colSpan={5} className="py-12 text-center text-slate-500">No studies yet</td></tr>
            ) : history.map((h, i) => (
              <tr key={h.study_id}>
                <td className="py-3 pr-4 text-slate-500">{i + 1}</td>
                <td className="py-3 pr-4 font-medium text-slate-800">{h.patient_name}</td>
                <td className="py-3 pr-4">
                  <span className={h.primary_label === 'Normal' ? 'badge-success' : h.primary_label === 'Pending' ? 'badge' : 'badge-danger'}>
                    {h.primary_label}
                  </span>
                </td>
                <td className="py-3 pr-4">
                  {h.confidence > 0 ? (
                    <div className="flex items-center gap-2">
                      <div className="w-20 h-2 bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full bg-cyan-600 rounded-full"
                          style={{ width: `${h.confidence * 100}%` }} />
                      </div>
                      <span className="text-slate-600">{(h.confidence * 100).toFixed(1)}%</span>
                    </div>
                  ) : '—'}
                </td>
                <td className="py-3 text-slate-500">{new Date(h.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
