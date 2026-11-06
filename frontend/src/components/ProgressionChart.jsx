import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend, Filler,
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

export default function ProgressionChart({ timeline }) {
  if (!timeline?.length) {
    return <p className="text-slate-400 text-sm text-center py-8">No scan history for progression tracking</p>
  }

  const labels = timeline.map((t) => new Date(t.date).toLocaleDateString())
  const confidence = timeline.map((t) => (t.confidence * 100).toFixed(1))
  const isAbnormal = timeline.map((t) => t.label !== 'Normal' ? 1 : 0)

  const data = {
    labels,
    datasets: [
      {
        label: 'AI Confidence %',
        data: confidence,
        borderColor: '#0284c7',
        backgroundColor: 'rgba(2,132,199,0.1)',
        fill: true,
        tension: 0.4,
        yAxisID: 'y',
      },
      {
        label: 'Abnormality Detected',
        data: isAbnormal.map(v => v * 100),
        borderColor: '#ef4444',
        borderDash: [5, 5],
        pointRadius: 6,
        fill: false,
        yAxisID: 'y',
      },
    ],
  }

  const options = {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'bottom' },
      title: { display: true, text: 'Disease Progression Over Time' },
    },
    scales: {
      y: { min: 0, max: 100, title: { display: true, text: '%' } },
    },
  }

  return (
    <div>
      <Line data={data} options={options} />
      <div className="mt-3 flex flex-wrap gap-2">
        {timeline.map((t) => (
          <span key={t.study_id} className={`text-xs px-2 py-1 rounded-full ${
            t.label === 'Normal' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
          }`}>
            {new Date(t.date).toLocaleDateString()}: {t.label}
          </span>
        ))}
      </div>
    </div>
  )
}
