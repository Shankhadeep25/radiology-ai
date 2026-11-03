import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js'
import { Doughnut, Bar } from 'react-chartjs-2'
import { dashboard } from '../api'
import PacsPanel from '../components/PacsPanel'

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement)

const chartDefaults = {
  plugins: { legend: { labels: { color: '#64748b' } } },
  scales: {
    x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(15,23,42,0.06)' } },
    y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(15,23,42,0.06)' } },
  },
}

const METRIC_THEMES = {
  blue: {
    bg: 'bg-blue-50/80 hover:bg-blue-50/95 border-blue-200/90 hover:border-blue-300 shadow-xs',
    label: 'text-blue-700',
    value: 'text-blue-950',
    sub: 'text-blue-600',
  },
  green: {
    bg: 'bg-emerald-50/80 hover:bg-emerald-50/95 border-emerald-200/90 hover:border-emerald-300 shadow-xs',
    label: 'text-emerald-700',
    value: 'text-emerald-950',
    sub: 'text-emerald-600',
  },
  orange: {
    bg: 'bg-amber-50/80 hover:bg-amber-50/95 border-amber-200/90 hover:border-amber-300 shadow-xs',
    label: 'text-amber-800',
    value: 'text-amber-950',
    sub: 'text-amber-700',
  },
  pink: {
    bg: 'bg-rose-50/80 hover:bg-rose-50/95 border-rose-200/90 hover:border-rose-300 shadow-xs',
    label: 'text-rose-700',
    value: 'text-rose-950',
    sub: 'text-rose-600',
  },
}

function MetricCard({ label, value, sub, variant = 'blue' }) {
  const theme = METRIC_THEMES[variant] || METRIC_THEMES.blue
  return (
    <div className={`rounded-2xl p-5 sm:p-6 border transition-colors flex flex-col justify-between ${theme.bg}`}>
      <div>
        <p className={`text-xs font-semibold uppercase tracking-wider ${theme.label}`}>{label}</p>
        <p className={`text-3xl sm:text-4xl font-bold mt-2 font-sans tracking-tight ${theme.value}`}>{value}</p>
      </div>
      <div className="mt-2.5 min-h-[1.125rem]">
        {sub && <p className={`text-xs font-medium ${theme.sub}`}>{sub}</p>}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null)
  const [history, setHistory] = useState([])

  useEffect(() => {
    dashboard.metrics().then((r) => setMetrics(r.data)).catch(() => setMetrics({
      total_studies: 0, total_patients: 0, pneumonia_cases: 0,
      normal_cases: 0, avg_confidence: 0, needs_review_count: 0,
    }))
    dashboard.history().then((r) => setHistory(r.data.slice(0, 5))).catch(() => {})
  }, [])

  if (!metrics) {
    return (
      <div className="flex justify-center py-24">
        <div className="w-10 h-10 rounded-full border-2 border-cyan-200 border-t-cyan-600 animate-spin" />
      </div>
    )
  }

  const pieData = {
    labels: ['Pneumonia', 'Normal', 'Other'],
    datasets: [{
      data: [metrics.pneumonia_cases, metrics.normal_cases,
        Math.max(0, metrics.total_studies - metrics.pneumonia_cases - metrics.normal_cases)],
      backgroundColor: ['rgba(244,63,94,0.85)', 'rgba(16,185,129,0.85)', 'rgba(148,163,184,0.7)'],
      borderWidth: 0,
    }],
  }

  const confidenceData = {
    labels: history.map((h) => h.patient_name.split(' ')[0]),
    datasets: [{
      label: 'Confidence %',
      data: history.map((h) => (h.confidence * 100).toFixed(1)),
      backgroundColor: 'rgba(8, 145, 178, 0.7)',
      borderRadius: 6,
      borderSkipped: false,
    }],
  }

  return (
    <div className="space-y-6 sm:space-y-8">
      <div className="hero-glass rounded-2xl p-6 sm:p-8">
        <div className="relative flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div>
            <p className="text-cyan-700 text-xs font-semibold uppercase tracking-[0.2em]">Clinical AI Decision Support</p>
            <h2 className="text-3xl sm:text-4xl font-sans font-bold tracking-tight text-slate-900 mt-1.5">Radiology AI Assistant</h2>
            <p className="text-slate-600 mt-2 max-w-xl leading-relaxed text-sm sm:text-base">
              Explainable chest X-ray analysis with Grad-CAM, differential diagnosis,
              and auto-generated radiology reports.
            </p>
          </div>
          <Link to="/analyze" className="btn-primary inline-flex items-center gap-2 px-6 py-3 self-start text-sm font-semibold rounded-xl">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Start Analysis
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
        <MetricCard label="Total Studies" value={metrics.total_studies} variant="blue" />
        <MetricCard label="Patients" value={metrics.total_patients} variant="green" />
        <MetricCard label="Avg Confidence" value={`${(metrics.avg_confidence * 100).toFixed(1)}%`} variant="orange" />
        <MetricCard label="Needs Review" value={metrics.needs_review_count} sub="Low confidence" variant="pink" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 sm:gap-6">
        <div className="card rounded-2xl p-6">
          <h3 className="font-sans font-bold text-slate-900 tracking-tight text-lg mb-4">Finding Distribution</h3>
          <div className="h-56 flex items-center justify-center">
            {metrics.total_studies > 0 ? (
              <Doughnut data={pieData} options={{ maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#64748b' } } } }} />
            ) : (
              <div className="text-center text-slate-500">
                <p>No studies yet</p>
                <Link to="/analyze" className="text-cyan-600 text-sm hover:underline mt-2 inline-block">Run your first analysis →</Link>
              </div>
            )}
          </div>
        </div>
        <div className="card rounded-2xl p-6">
          <h3 className="font-sans font-bold text-slate-900 tracking-tight text-lg mb-4">Confidence Scores</h3>
          <div className="h-56">
            {history.length > 0 ? (
              <Bar data={confidenceData} options={{
                ...chartDefaults,
                maintainAspectRatio: false,
                scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, max: 100, ticks: { ...chartDefaults.scales.y.ticks, callback: (v) => v + '%' } } },
                plugins: { legend: { display: false } },
              }} />
            ) : (
              <div className="h-full flex items-center justify-center text-slate-500">No data yet</div>
            )}
          </div>
        </div>
      </div>

      <PacsPanel />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-5">
        {[
          { title: 'X-Ray MVP', desc: 'DenseNet121 pneumonia + 12 chest findings with Grad-CAM', icon: '🫁', color: 'from-cyan-50 to-blue-50 border-cyan-100' },
          { title: 'CT / MRI / US', desc: 'Multi-modality pipelines for nodules, tumors, BI-RADS', icon: '🧠', color: 'from-violet-50 to-purple-50 border-violet-100' },
          { title: 'PACS + FHIR', desc: 'Orthanc, DICOMweb, HL7 FHIR integration', icon: '🏥', color: 'from-pink-50 to-rose-50 border-pink-100' },
        ].map((f) => (
          <div key={f.title} className={`card rounded-2xl p-6 text-center bg-gradient-to-br ${f.color}`}>
            <div className="text-4xl mb-3 select-none">{f.icon}</div>
            <h4 className="font-sans font-semibold text-slate-900 text-base">{f.title}</h4>
            <p className="text-sm text-slate-600 mt-2">{f.desc}</p>
          </div>
        ))}
      </div>

      {history.length > 0 && (
        <div className="card rounded-2xl p-6">
          <h3 className="font-sans font-bold text-slate-900 tracking-tight text-lg mb-4">Recent Studies</h3>
          <table className="w-full text-sm table-glass">
            <thead>
              <tr>
                <th className="pb-3 text-left">Patient</th>
                <th className="pb-3 text-left">Finding</th>
                <th className="pb-3 text-left">Confidence</th>
                <th className="pb-3 text-left">Date</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h) => (
                <tr key={h.study_id}>
                  <td className="py-3 font-medium text-slate-800">{h.patient_name}</td>
                  <td className="py-3">
                    <span className={h.primary_label === 'Normal' ? 'badge-success' : 'badge-danger'}>
                      {h.primary_label}
                    </span>
                  </td>
                  <td className="py-3 text-slate-600">{(h.confidence * 100).toFixed(1)}%</td>
                  <td className="py-3 text-slate-500">{new Date(h.created_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
