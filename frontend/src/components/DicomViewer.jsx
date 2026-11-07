import { useEffect, useRef, useState } from 'react'

async function fetchStudyImage(imageUrl) {
  const token = localStorage.getItem('token')
  const headers = token ? { Authorization: `Bearer ${token}` } : {}
  const response = await fetch(imageUrl, { headers })
  if (!response.ok) {
    throw new Error(`Could not load scan (HTTP ${response.status})`)
  }
  const buffer = await response.arrayBuffer()
  const contentType = response.headers.get('content-type') || 'image/jpeg'
  return { buffer, contentType }
}

export default function DicomViewer({ studyId, fileType, imageUrl, patientName }) {
  const viewportRef = useRef(null)
  const blobRef = useRef(null)
  const [status, setStatus] = useState('loading')
  const [displayUrl, setDisplayUrl] = useState(null)
  const [mode, setMode] = useState('image') // 'image' | 'dicom'
  const [error, setError] = useState('')

  useEffect(() => {
    if (!imageUrl) {
      setStatus('error')
      setError('No image URL for this study')
      return
    }

    let active = true

    async function load() {
      setStatus('loading')
      setError('')
      setDisplayUrl(null)
      setMode('image')

      try {
        const { buffer, contentType } = await fetchStudyImage(imageUrl)
        const isDicom = fileType === 'dicom' || contentType.includes('dicom')

        if (!isDicom) {
          const blob = new Blob([buffer], { type: contentType })
          const url = URL.createObjectURL(blob)
          blobRef.current = url
          if (active) {
            setDisplayUrl(url)
            setMode('image')
            setStatus('ready')
          }
          return
        }

        // DICOM via Cornerstone
        const dicomParser = (await import('dicom-parser')).default
        const cornerstone = (await import('cornerstone-core')).default
        const cornerstoneMath = (await import('cornerstone-math')).default
        const cornerstoneWADO = (await import('cornerstone-wado-image-loader')).default

        cornerstoneWADO.external.cornerstone = cornerstone
        cornerstoneWADO.external.dicomParser = dicomParser
        cornerstoneWADO.external.cornerstoneMath = cornerstoneMath
        cornerstoneWADO.configure({ useWebWorkers: false })

        const blob = new Blob([buffer], { type: 'application/dicom' })
        const url = URL.createObjectURL(blob)
        blobRef.current = url

        const element = viewportRef.current
        if (!element || !active) return

        element.innerHTML = ''
        const canvasHost = document.createElement('div')
        canvasHost.style.width = '100%'
        canvasHost.style.height = '100%'
        element.appendChild(canvasHost)

        cornerstone.enable(canvasHost)
        const image = await cornerstone.loadImage(`wadouri:${url}`)
        if (!active) return
        cornerstone.displayImage(canvasHost, image)
        setMode('dicom')
        setStatus('ready')
      } catch (err) {
        console.error('Viewer error:', err)
        if (!active) return
        // Fallback: try showing as generic image blob
        try {
          const { buffer, contentType } = await fetchStudyImage(imageUrl)
          const blob = new Blob([buffer], { type: contentType })
          const url = URL.createObjectURL(blob)
          blobRef.current = url
          setDisplayUrl(url)
          setMode('image')
          setStatus('ready')
        } catch {
          setStatus('error')
          setError(err.message || 'Failed to load medical image')
        }
      }
    }

    load()

    return () => {
      active = false
      if (blobRef.current) {
        URL.revokeObjectURL(blobRef.current)
        blobRef.current = null
      }
      if (viewportRef.current) {
        viewportRef.current.innerHTML = ''
      }
    }
  }, [imageUrl, fileType, studyId])

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-slate-800">Medical Image Viewer</h4>
        <div className="flex gap-2">
          <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full uppercase">
            {fileType || 'image'}
          </span>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
            status === 'ready' ? 'bg-green-100 text-green-700' :
            status === 'error' ? 'bg-red-100 text-red-700' :
            'bg-amber-100 text-amber-700'
          }`}>
            {status === 'ready' ? 'Loaded' : status === 'error' ? 'Error' : 'Loading...'}
          </span>
        </div>
      </div>

      <div
        className="bg-black rounded-xl overflow-hidden relative flex items-center justify-center"
        style={{ height: 300, minHeight: 300 }}
      >
        {status === 'loading' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center z-20 gap-2">
            <div className="animate-spin rounded-full h-9 w-9 border-2 border-white border-t-transparent" />
            <span className="text-white/70 text-xs">Loading scan...</span>
          </div>
        )}

        {status === 'error' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center z-20 p-6 text-center">
            <p className="text-red-300 text-sm">{error}</p>
            <p className="text-white/50 text-xs mt-2">Study #{studyId}</p>
          </div>
        )}

        {mode === 'image' && displayUrl && status === 'ready' && (
          <img
            src={displayUrl}
            alt={patientName ? `Scan for ${patientName}` : 'Medical scan'}
            className="max-w-full max-h-full w-auto h-auto object-contain"
            style={{ maxHeight: 300 }}
            onError={() => {
              if (displayUrl !== imageUrl) {
                setDisplayUrl(imageUrl)
              } else {
                setStatus('error')
                setError('Failed to render medical scan image')
              }
            }}
          />
        )}

        {mode === 'dicom' && (
          <div ref={viewportRef} className="absolute inset-0 w-full h-full" />
        )}

        {mode === 'image' && !displayUrl && status === 'ready' && (
          <div ref={viewportRef} className="absolute inset-0 w-full h-full" />
        )}
      </div>

      {studyId && status === 'ready' && (
        <p className="text-xs text-slate-400">
          Study #{studyId}
          {mode === 'dicom' ? ' · Cornerstone.js DICOM renderer' : ' · Authenticated secure image stream'}
        </p>
      )}
    </div>
  )
}
