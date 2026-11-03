const URGENCY_COLORS = {
  critical: 'border-rose-300 text-rose-700 bg-rose-50',
  high: 'border-orange-300 text-orange-700 bg-orange-50',
  medium: 'border-amber-300 text-amber-700 bg-amber-50',
  none: 'border-emerald-300 text-emerald-700 bg-emerald-50',
}

const STATUS_LABELS = {
  detected: { text: 'Detected', class: 'bg-rose-600 text-white' },
  possible: { text: 'Possible', class: 'bg-amber-500 text-white' },
}

export default function FindingsPanel({ findings, primaryLabel }) {
  if (!findings?.length) return null

  return (
    <div className="card">
      <h3 className="font-display font-semibold text-slate-900 mb-1">All Detected Conditions</h3>
      <p className="text-xs text-slate-500 mb-4">
        AI screened for {findings.length}+ chest pathologies
      </p>

      <div className="space-y-2">
        {findings.map((f, i) => {
          const isPrimary = f.condition === primaryLabel
          const status = STATUS_LABELS[f.status] || STATUS_LABELS.possible
          const urgency = URGENCY_COLORS[f.urgency] || URGENCY_COLORS.medium

          return (
            <div key={f.condition}
              className={`flex items-center gap-3 p-3 rounded-xl border transition-colors ${
                isPrimary ? 'border-cyan-300 bg-cyan-50/70' : 'border-slate-200 bg-slate-50'
              }`}>
              <span className="text-sm font-semibold w-6 text-center text-slate-400">{i + 1}</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-medium text-slate-800">{f.condition}</span>
                  {isPrimary && (
                    <span className="text-xs bg-cyan-600 text-white px-2 py-0.5 rounded-full font-medium">Primary</span>
                  )}
                  <span className={`text-xs px-2 py-0.5 rounded-full ${status.class}`}>{status.text}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full border ${urgency}`}>{f.category}</span>
                </div>
                <div className="flex items-center gap-2 mt-1.5">
                  <div className="flex-1 h-2 bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${f.status === 'detected' ? 'bg-rose-500' : 'bg-amber-500'}`}
                      style={{ width: `${f.probability * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-semibold text-slate-600 w-12 text-right">
                    {(f.probability * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
