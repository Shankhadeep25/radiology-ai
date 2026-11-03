import { useEffect, useState } from 'react'
import { patients } from '../api'
import ProgressionChart from '../components/ProgressionChart'

export default function Patients() {
  const [patientList, setPatientList] = useState([])
  const [selected, setSelected] = useState(null)
  const [timeline, setTimeline] = useState([])

  useEffect(() => {
    patients.list().then((r) => setPatientList(r.data))
  }, [])

  const selectPatient = async (patient) => {
    setSelected(patient)
    const { data } = await patients.timeline(patient.id)
    setTimeline(data)
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="page-title">Patients</h2>
        <p className="page-subtitle">Patient records and disease progression tracking</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card lg:col-span-1">
          <h3 className="font-display font-semibold text-slate-900 mb-4">All Patients ({patientList.length})</h3>
          {patientList.length === 0 ? (
            <p className="text-slate-500 text-sm">No patients yet — run an analysis first</p>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {patientList.map((p) => (
                <button key={p.id} onClick={() => selectPatient(p)}
                  className={`w-full text-left p-3 rounded-xl border transition-colors ${
                    selected?.id === p.id
                      ? 'border-cyan-400 bg-cyan-50/70'
                      : 'border-slate-200 bg-slate-50 hover:bg-white'
                  }`}>
                  <p className="font-medium text-slate-800">{p.name}</p>
                  <p className="text-xs text-slate-500">
                    {p.age ? `Age ${p.age}` : ''} {p.gender ? `· ${p.gender}` : ''}
                  </p>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="card lg:col-span-2">
          {!selected ? (
            <div className="h-64 flex items-center justify-center text-slate-500">
              Select a patient to view progression timeline
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <h3 className="font-display font-semibold text-slate-900 text-lg">{selected.name}</h3>
                <p className="text-sm text-slate-500">{timeline.length} scan(s) on record</p>
              </div>
              <ProgressionChart timeline={timeline} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
