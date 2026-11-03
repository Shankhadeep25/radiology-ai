const STEPS = [
  { id: 1, label: 'Preprocessing', desc: 'CLAHE, denoise, normalize' },
  { id: 2, label: 'AI Inference', desc: 'DenseNet121 analysis' },
  { id: 3, label: 'Grad-CAM', desc: 'Generating heatmap' },
  { id: 4, label: 'Report', desc: 'Clinical findings' },
  { id: 5, label: 'Complete', desc: 'Results ready' },
]

export default function PipelineSteps({ currentStep }) {
  return (
    <div className="card bg-gradient-to-r from-cyan-50 to-blue-50 border-cyan-200">
      <div className="flex items-center justify-between gap-2 overflow-x-auto pb-1">
        {STEPS.map((step, i) => {
          const done = currentStep >= step.id
          const active = currentStep === step.id

          return (
            <div key={step.id} className="flex items-center gap-2 flex-1 min-w-0">
              <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold transition-colors ${
                done ? 'bg-cyan-600 text-white shadow-xs' :
                active ? 'bg-cyan-100 text-cyan-800 border border-cyan-400' :
                'bg-slate-100 text-slate-400 border border-slate-200'
              }`}>
                {done && currentStep > step.id ? '✓' : step.id}
              </div>

              <div className="min-w-0 hidden sm:block">
                <p className={`text-sm font-medium truncate ${done ? 'text-cyan-800' : 'text-slate-500'}`}>
                  {step.label}
                </p>
                <p className="text-xs text-slate-400 truncate">{step.desc}</p>
              </div>

              {i < STEPS.length - 1 && (
                <div className={`flex-1 h-0.5 mx-1 rounded-full transition-colors ${
                  done ? 'bg-cyan-500' : 'bg-slate-200'
                }`} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
