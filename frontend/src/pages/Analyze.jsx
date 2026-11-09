import { useState, useEffect } from 'react'
import { patients, studies, dashboard } from '../api'
import ConfidenceMeter from '../components/ConfidenceMeter'
import DifferentialList from '../components/DifferentialList'
import FindingsPanel from '../components/FindingsPanel'
import PipelineSteps from '../components/PipelineSteps'
import DicomViewer from '../components/DicomViewer'
import PacsPanel from '../components/PacsPanel'
import { useStudyChat } from '../context/StudyChatContext'

const MODALITIES = [
  { id: 'XR', label: 'X-Ray (MVP)', desc: 'DenseNet121 + Grad-CAM (18 pathologies)' },
  { id: 'CT', label: 'CT Chest (Baseline)', desc: 'DenseNet121 research baseline (not clinically validated)' },
  { id: 'MR', label: 'MRI Brain (Baseline)', desc: 'DenseNet121 research baseline (not clinically validated)' },
  { id: 'US', label: 'Ultrasound (Baseline)', desc: 'DenseNet121 research baseline (not clinically validated)' },
]

const DEMO_CASES = [
  { name: 'Rajesh Kumar', age: 58, sample: 'pneumonia_demo.jpg', label: 'X-Ray Pneumonia', modality: 'XR' },
  { name: 'Priya Sharma', age: 34, sample: 'normal_demo.jpg', label: 'X-Ray Normal', modality: 'XR' },
  { name: 'Amit Singh', age: 62, sample: 'ct_nodule_demo.jpg', label: 'CT Adenocarcinoma', modality: 'CT' },
  { name: 'Meera Iyer', age: 47, sample: 'mri_tumor_demo.jpg', label: 'MRI Glioma', modality: 'MR' },
  { name: 'Ananya Rao', age: 41, sample: 'us_breast_demo.jpg', label: 'Ultrasound Benign', modality: 'US' },
]

export default function Analyze() {
  const [patientName, setPatientName] = useState('')
  const [patientAge, setPatientAge] = useState('')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [pipelineStep, setPipelineStep] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [demoSamples, setDemoSamples] = useState([])
  const [modality, setModality] = useState('XR')
  const { setStudyContext } = useStudyChat()

  useEffect(() => {
    dashboard.demoSamples?.() ||
      fetch('/api/v1/demo/samples').then(r => r.json()).then(setDemoSamples).catch(() => {})
  }, [])

  useEffect(() => {
    if (result?.prediction) {
      setStudyContext({
        patient_name: patientName,
        patient_age: patientAge,
        primary_label: result.prediction.primary_label,
        confidence: result.prediction.confidence,
        severity: result.prediction.severity,
        modality,
        telemetry: result.prediction.volumetrics,
        findings: result.report?.findings,
        impression: result.report?.impression,
        all_findings: result.prediction.differential_diagnosis,
      })
    }
  }, [result, modality, patientName, patientAge, setStudyContext])

  const handleFileChange = (e) => {
    const f = e.target.files[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError('')
  }

  const loadDemoCase = async (demoCase) => {
    setPatientName(demoCase.name)
    setPatientAge(String(demoCase.age))
    setModality(demoCase.modality || 'XR')
    setError('')
    setResult(null)
    try {
      const token = localStorage.getItem('token')
      let res = await fetch(`/api/v1/demo/samples/${demoCase.sample}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      })
      if (!res.ok) {
        res = await fetch(`/demo_samples/${demoCase.sample}`)
      }
      if (!res.ok) throw new Error('Could not load clinical demo sample file')
      const blob = await res.blob()
      const f = new File([blob], demoCase.sample, { type: blob.type || 'image/jpeg' })
      setFile(f)
      setPreview(URL.createObjectURL(blob))
    } catch (err) {
      setError(`Failed to load ${demoCase.label}: ${err.message}`)
    }
  }

  const handleAnalyze = async () => {
    if (!file || !patientName) {
      setError('Patient name and scan file are required')
      return
    }
    setLoading(true)
    setError('')
    setPipelineStep(1)
    setResult(null)

    try {
      setPipelineStep(1) // Step 1: Preprocessing & Normalization
      const { data: patient } = await patients.create({
        name: patientName,
        age: patientAge ? parseInt(patientAge) : null,
      })

      setPipelineStep(2) // Step 2: Uploading & Modality Routing
      const { data: upload } = await studies.upload(patient.id, file, modality)

      setPipelineStep(3) // Step 3: Neural Network & Feature Inference
      const { data: analysis } = await studies.predict(upload.study_id)

      setPipelineStep(4) // Step 4: Grad-CAM & Telemetry Extraction
      await new Promise(r => setTimeout(r, 100))

      setPipelineStep(5) // Step 5: Clinical Report Finalization
      setResult(analysis)
    } catch (err) {
      const status = err.response?.status
      const detail = err.response?.data?.detail
      if (status === 503) {
        setError(detail || 'This model is not enabled. Baseline models require ENABLE_BASELINE_MODELS=1.')
      } else {
        setError(detail || 'Analysis failed. Is the backend running on port 8000?')
      }
    } finally {
      setLoading(false)
    }
  }

  const prediction = result?.prediction
  const report = result?.report
  const study = result?.study
  const tel = prediction?.volumetrics || {}
  const isBaseline = prediction?.status === 'research_baseline' || ['CT', 'MR', 'US'].includes(modality)

  return (
    <div className="space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4">
        <div>
          <h2 className="page-title">Multi-Modality AI Diagnostic System</h2>
          <p className="page-subtitle">DenseNet121 Chest Radiography · 3D CT Volumetrics · MRI Symmetry · US BI-RADS</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider mr-1">Demo Scans:</span>
          {DEMO_CASES.map((d) => (
            <button
              key={d.sample}
              onClick={() => loadDemoCase(d)}
              className="text-xs font-medium px-3 py-1.5 rounded-lg border border-slate-200 bg-white hover:border-cyan-400 hover:text-cyan-600 transition-colors shadow-xs">
              {d.label}
            </button>
          ))}
        </div>
      </div>

      <PipelineSteps currentStep={pipelineStep} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card space-y-5">
          <h3 className="font-display font-semibold text-slate-900 border-b border-slate-100 pb-3">Patient & Study Intake</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Patient Name</label>
              <input type="text" value={patientName} onChange={(e) => setPatientName(e.target.value)}
                placeholder="e.g. Ramesh Chandra" className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Patient Age</label>
              <input type="number" value={patientAge} onChange={(e) => setPatientAge(e.target.value)}
                placeholder="e.g. 52" className="input-field" />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Imaging Modality</label>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {MODALITIES.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setModality(m.id)}
                  className={`p-3 rounded-xl border text-left transition-colors ${
                    modality === m.id
                      ? 'border-cyan-500 bg-cyan-50/70 shadow-xs ring-1 ring-cyan-500/30'
                      : 'border-slate-200 bg-slate-50 hover:border-cyan-300'
                  }`}>
                  <p className="font-semibold text-slate-800">{m.label}</p>
                  <p className="text-slate-500 mt-0.5">{m.desc}</p>
                </button>
              ))}
            </div>
            {modality !== 'XR' && (
              <div className="mt-2.5 p-3 bg-red-50 border-2 border-red-500 rounded-xl text-red-700 text-xs font-semibold flex items-start gap-2 shadow-xs">
                <span className="text-base leading-none select-none">⚠️</span>
                <span className="leading-relaxed">
                  RESEARCH BASELINE: NOT CLINICALLY VALIDATED. Not for diagnosis or patient care. Trained on one public dataset. Probabilities are not calibrated. Wrong-modality images can still receive confident answers.
                </span>
              </div>
            )}
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Scan (PNG / JPG / DICOM)</label>
            <div className={`border-2 border-dashed rounded-2xl p-4 text-center transition-colors ${
              preview ? 'border-cyan-400 bg-cyan-50/50' : 'border-slate-300 hover:border-cyan-400 bg-slate-50'
            }`}>
              <input type="file" accept=".png,.jpg,.jpeg,.dcm,.dicom" onChange={handleFileChange}
                className="hidden" id="file-upload" />
              <label htmlFor="file-upload" className="cursor-pointer block">
                {preview ? (
                  <img src={preview} alt="Preview" className="max-h-56 mx-auto rounded-xl shadow-xs" />
                ) : (
                  <div className="text-slate-500 py-8">
                    <svg className="w-12 h-12 mx-auto mb-2 text-cyan-400/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <p className="font-medium text-slate-600">Drop scan or click to upload</p>
                    <p className="text-xs mt-1">PNG, JPG, DICOM supported</p>
                  </div>
                )}
              </label>
            </div>
          </div>
          {error && (
            <div className="bg-rose-50 border border-rose-200 text-rose-800 text-sm rounded-xl p-3.5 flex items-start gap-2.5 animate-fade-in">
              <span className="text-lg leading-none">🚫</span>
              <div>
                <p className="font-semibold text-rose-900">{error.includes('rejected') || error.includes('Invalid') || error.includes('photograph') ? 'Authenticity Guard: Image Rejected' : 'Validation Error'}</p>
                <p className="text-xs text-rose-700 mt-0.5 leading-relaxed">{error}</p>
              </div>
            </div>
          )}
          <button onClick={handleAnalyze} disabled={loading || !file}
            className="btn-primary w-full py-3 text-base">
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Executing AI Diagnostic Pipeline...
              </span>
            ) : 'Run AI Analysis'}
          </button>
        </div>

        <div className="space-y-4">
          {!result ? (
            <div className="card min-h-[400px] flex items-center justify-center">
              <div className="text-center text-slate-500 max-w-xs">
                <div className="w-20 h-20 bg-gradient-to-br from-cyan-100 to-violet-100 rounded-2xl flex items-center justify-center mx-auto mb-4 border border-slate-200">
                  <svg className="w-10 h-10 text-cyan-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <p className="font-medium text-slate-600">AI diagnostic results will appear here</p>
                <p className="text-sm mt-2">Select any scan above or upload your own to begin</p>
              </div>
            </div>
          ) : (
            <>
              <div className={`card card-glow ${
                isBaseline
                  ? 'border-2 border-red-500 bg-red-50/20'
                  : prediction.primary_label.includes('Normal') || prediction.primary_label.includes('No Acute') || prediction.primary_label.includes('Negative')
                  ? 'border-emerald-300 bg-emerald-50'
                  : 'border-rose-300 bg-rose-50'
              }`}>
                {isBaseline && (
                  <div className="bg-red-600 text-white p-4 rounded-xl font-bold text-sm tracking-wide shadow-md mb-4 flex items-start gap-2.5">
                    <span className="text-xl leading-none select-none">⚠️</span>
                    <div>
                      <div className="uppercase tracking-wider font-extrabold text-sm mb-1">
                        RESEARCH BASELINE: NOT CLINICALLY VALIDATED
                      </div>
                      <div className="text-xs font-normal text-red-100 leading-relaxed">
                        Not for diagnosis or patient care. Trained on one public dataset. Probabilities are not calibrated. Wrong-modality images can still receive confident answers.
                      </div>
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-display font-semibold text-slate-900">
                    {isBaseline ? 'Research Baseline Prediction' : 'AI Clinical Prediction'}
                  </h3>
                  <span className="text-xs text-slate-500 font-mono">{prediction.model_version?.split('(')[0]}</span>
                </div>
                <div className="flex flex-wrap items-center gap-3 mb-4">
                  <div className={`text-3xl font-display font-bold ${
                    isBaseline
                      ? 'text-slate-900'
                      : prediction.primary_label.includes('Normal') || prediction.primary_label.includes('No Acute') || prediction.primary_label.includes('Negative')
                      ? 'text-emerald-600'
                      : 'text-rose-600'
                  }`}>{prediction.primary_label}</div>

                  {!isBaseline ? (
                    <span className={
                      prediction.uncertainty_level === 'high' ? 'badge-success' :
                      prediction.uncertainty_level === 'medium' ? 'badge-warn' : 'badge-danger'
                    }>
                      {prediction.uncertainty_level === 'needs_review' ? 'Needs Human Review' :
                       `${prediction.uncertainty_level} confidence`}
                    </span>
                  ) : (
                    <span className="px-2.5 py-1 text-xs font-semibold rounded-lg bg-slate-200 text-slate-700">
                      Uncalibrated Score: {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  )}
                </div>

                {!isBaseline && <ConfidenceMeter confidence={prediction.confidence} />}

                {isBaseline && prediction.model_metrics && (
                  <div className="bg-white/95 border border-red-200 rounded-xl p-3 mb-4 text-xs">
                    <div className="font-semibold text-slate-800 uppercase tracking-wider text-[11px] mb-2 border-b border-slate-100 pb-1">
                      Model Benchmark Performance (From Metrics JSON)
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-slate-700">
                      <div>
                        <span className="text-slate-500 font-medium">Test Accuracy: </span>
                        <span className="font-mono font-bold">
                          {prediction.model_metrics.accuracy !== null && prediction.model_metrics.accuracy !== undefined
                            ? (prediction.model_metrics.accuracy * 100).toFixed(1) + '%'
                            : 'null'}
                        </span>
                        {prediction.model_metrics.accuracy_ci && (
                          <span className="text-slate-400 text-[10px] ml-1">
                            (95% CI: {(prediction.model_metrics.accuracy_ci[0] * 100).toFixed(1)}%–{(prediction.model_metrics.accuracy_ci[1] * 100).toFixed(1)}%)
                          </span>
                        )}
                      </div>
                      <div>
                        <span className="text-slate-500 font-medium">Macro AUC: </span>
                        <span className="font-mono font-bold">
                          {prediction.model_metrics.macro_auc !== null && prediction.model_metrics.macro_auc !== undefined
                            ? prediction.model_metrics.macro_auc.toFixed(3)
                            : 'null'}
                        </span>
                        {prediction.model_metrics.macro_auc_ci && (
                          <span className="text-slate-400 text-[10px] ml-1">
                            (95% CI: {prediction.model_metrics.macro_auc_ci[0].toFixed(3)}–{prediction.model_metrics.macro_auc_ci[1].toFixed(3)})
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {isBaseline && prediction.all_scores && (
                  <div className="bg-white/95 border border-slate-200 rounded-xl p-3 mb-4">
                    <div className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                      All Class Probabilities
                    </div>
                    <div className="space-y-1.5">
                      {Object.entries(prediction.all_scores).map(([cls, score]) => (
                        <div key={cls} className="flex items-center justify-between text-xs">
                          <span className={`font-medium ${cls === prediction.primary_label ? 'text-slate-900 font-bold' : 'text-slate-600'}`}>{cls}</span>
                          <div className="flex items-center gap-2">
                            <div className="w-24 bg-slate-100 rounded-full h-2 overflow-hidden border border-slate-200">
                              <div className="bg-slate-700 h-2 rounded-full" style={{ width: `${Math.min(100, Math.max(0, score * 100))}%` }} />
                            </div>
                            <span className="font-mono text-slate-800 w-12 text-right">{(score * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {isBaseline && prediction.modality_warning && (
                  <div className="mt-2 text-xs bg-amber-50 border border-amber-300 text-amber-900 rounded-xl p-2.5">
                    ⚠️ {prediction.modality_warning}
                  </div>
                )}

                {!isBaseline && tel?.anatomical_location && (
                  <div className="mt-4 flex flex-wrap items-center gap-2 text-xs">
                    <span className="bg-sky-100 text-sky-800 border border-sky-300 rounded-lg px-2.5 py-1 font-semibold flex items-center gap-1 select-none">
                      📍 {tel.anatomical_location}
                    </span>
                    {tel.diameter_mm > 0 && (
                      <span className="bg-slate-100 text-slate-800 border border-slate-300 rounded-lg px-2.5 py-1 font-medium select-none">
                        📏 ~{tel.diameter_mm} mm span
                    </span>
                    )}
                    {tel.volume_mm3 > 0 && (
                      <span className="bg-slate-100 text-slate-800 border border-slate-300 rounded-lg px-2.5 py-1 font-medium select-none">
                        🧊 {tel.volume_mm3} mm³ ({tel.mass_g} g mass)
                    </span>
                    )}
                    {tel.volume_ml > 0 && (
                      <span className="bg-slate-100 text-slate-800 border border-slate-300 rounded-lg px-2.5 py-1 font-medium select-none">
                        🧊 {tel.volume_ml} mL volume ({tel.mass_g} g)
                    </span>
                    )}
                    {tel.birads && (
                      <span className="bg-purple-100 text-purple-800 border border-purple-300 rounded-lg px-2.5 py-1 font-bold select-none">
                        🏷️ {tel.birads}
                    </span>
                    )}
                  </div>
                )}

                {prediction?.modality_validation?.warning && (
                  <div className="mt-4 text-sm bg-amber-50 border border-amber-200 text-amber-800 rounded-xl px-4 py-3">
                    {prediction.modality_validation.warning}
                  </div>
                )}
                {(prediction.primary_label === 'Modality Mismatch' || prediction.primary_label === 'Invalid Image Rejected') && (
                  <div className="mt-4 text-sm bg-rose-50 border border-rose-200 text-rose-800 rounded-xl px-4 py-3">
                    {prediction?.rejection_reason
                      || prediction?.modality_validation?.warning
                      || prediction?.report_extra
                      || 'The uploaded scan does not match the selected modality or was rejected as a non-medical image.'}
                  </div>
                )}
              </div>

              {prediction.heatmap_url && (
                <div className="card">
                  <h3 className="font-display font-semibold text-slate-900 mb-1">Explainability — Neural Grad-CAM</h3>
                  <p className="text-xs text-slate-500 mb-3">Targeted backpropagation highlights image regions that triggered this diagnosis</p>
                  <img src={prediction.heatmap_url} alt="Grad-CAM"
                    className="w-full rounded-xl border border-slate-200 shadow-xs" />
                </div>
              )}

              {study && (
                <div className="card">
                  <DicomViewer
                    studyId={study.id}
                    fileType={study.file_type}
                    imageUrl={study.image_url}
                    patientName={patientName}
                  />
                </div>
              )}

              {!isBaseline && ((prediction.all_findings?.length > 0) ? (
                <FindingsPanel findings={prediction.all_findings} primaryLabel={prediction.primary_label} />
              ) : prediction.differential_diagnosis?.length > 0 && (
                <div className="card">
                  <h3 className="font-display font-semibold text-slate-900 mb-3">Differential Diagnosis</h3>
                  <DifferentialList items={prediction.differential_diagnosis} />
                </div>
              ))}

              {report && (
                <div className="card bg-slate-50">
                  {isBaseline && (
                    <div className="p-3 bg-red-100 border-2 border-red-500 text-red-900 rounded-xl text-xs font-bold uppercase tracking-wider mb-4 flex items-center gap-2">
                      <span>⚠️</span>
                      <span>RESEARCH BASELINE: NOT CLINICALLY VALIDATED. Not for diagnosis or patient care.</span>
                    </div>
                  )}
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="font-display font-semibold text-slate-900">Dynamic Clinical Report</h3>
                      <p className="text-xs text-slate-500">Synthesized with quantitative telemetry & guidelines</p>
                    </div>
                    {report.pdf_url && (
                      <a href={report.pdf_url} target="_blank" rel="noopener noreferrer"
                        className="btn-primary text-sm shadow-xs flex items-center gap-1.5">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        Download PDF
                      </a>
                    )}
                  </div>
                  <div className="space-y-4 text-sm bg-white rounded-xl p-4 border border-slate-200">
                    <div>
                      <p className="font-semibold text-cyan-700 uppercase text-xs tracking-widest">Findings</p>
                      <div className="text-slate-700 mt-2 leading-relaxed" dangerouslySetInnerHTML={{ __html: report.findings }} />
                    </div>
                    <div className="border-t border-slate-200 pt-4">
                      <p className="font-semibold text-violet-700 uppercase text-xs tracking-widest">Impression & Recommendations</p>
                      <div className="text-slate-700 mt-2 leading-relaxed" dangerouslySetInnerHTML={{ __html: report.impression }} />
                    </div>
                  </div>
                  <p className="text-xs text-slate-500 mt-3 italic">
                    Clinical decision support summary. Final interpretation by licensed radiologist required.
                  </p>
                </div>
              )}

              <PacsPanel />
            </>
          )}
        </div>
      </div>
    </div>
  )
}
