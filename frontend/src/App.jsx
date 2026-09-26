import { useCallback, useEffect, useRef, useState } from 'react'
import Sidebar from './components/Sidebar.jsx'
import TabGan from './components/TabGan.jsx'
import { TabOverview, TabEnhance, TabColorize, TabAnalysis, TabReport } from './components/Tabs.jsx'
import { Empty, Note, Progress } from './components/ui.jsx'
import { getHealth, getSamples, runGan, runPipeline } from './api.js'

const TABS = [
  { id: 'overview', label: 'Overview', icon: '▦' },
  { id: 'enhance', label: 'Enhancement', icon: '◐' },
  { id: 'colorize', label: 'Colorization', icon: '◑' },
  { id: 'analysis', label: 'Analysis', icon: '▣' },
  { id: 'report', label: 'Report', icon: '▤' },
  { id: 'gan', label: 'GAN Enhancer', icon: '◈' },
]

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [samples, setSamples] = useState([])
  const [selectedSample, setSelectedSample] = useState(null)
  const [health, setHealth] = useState(null)
  const [result, setResult] = useState(null)
  const [gan, setGan] = useState(null)
  const [busy, setBusy] = useState(null)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)
  const [ganError, setGanError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [notice, setNotice] = useState(null)
  const resultRef = useRef(null)

  // Object URL for the local preview; revoked on change to avoid leaks.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const pollHealth = useCallback(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: 'down' }))
  }, [])

  useEffect(() => {
    pollHealth()
    const t = setInterval(pollHealth, 15000)
    return () => clearInterval(t)
  }, [pollHealth])

  useEffect(() => {
    getSamples().then((d) => setSamples(d.samples || [])).catch(() => setSamples([]))
  }, [])

  // A fresh upload clears stale results so the UI never shows the previous image.
  const handleFile = useCallback((f) => {
    setFile(f)
    setResult(null)
    setGan(null)
    setError(null)
    setGanError(null)
    setNotice(null)
  }, [])

  const handleSample = useCallback(async (s) => {
    if (!s) return
    try {
      const res = await fetch(s.url)
      const blob = await res.blob()
      const f = new File([blob], s.name, { type: blob.type || 'image/png' })
      setSelectedSample(s.name)
      handleFile(f)
    } catch {
      setError(`Could not load sample "${s.name}".`)
    }
  }, [handleFile])

  const handleRunPipeline = useCallback(async () => {
    if (!file) return
    setBusy('pipeline')
    setError(null)
    setNotice(null)
    setProgress(15)
    const tick = setInterval(() => setProgress((p) => Math.min(p + 7, 88)), 400)
    try {
      const data = await runPipeline(file)
      setResult(data)
      setProgress(100)
      setTab('overview')
      setNotice('Pipeline complete. Results below.')
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60)
    } catch (e) {
      setError(e.message || 'Pipeline failed.')
    } finally {
      clearInterval(tick)
      setTimeout(() => setProgress(0), 700)
      setBusy(null)
    }
  }, [file])

  const handleRunGan = useCallback(async (opts = {}) => {
    if (!file) return
    setBusy('gan')
    setGanError(null)
    try {
      const data = await runGan(file, {
        scale: opts.scale ?? 4,
        tile: opts.tile ?? 0,
        device: opts.device ?? 'auto',
      })
      setGan(data)
    } catch (e) {
      setGanError(e.message || 'GAN enhancement failed.')
    } finally {
      setBusy(null)
    }
  }, [file])

  const needsResult = ['overview', 'enhance', 'colorize', 'analysis', 'report']

  return (
    <div className="app">
      <Sidebar
        file={file}
        previewUrl={previewUrl}
        onFile={handleFile}
        samples={samples}
        selectedSample={selectedSample}
        onSelectSample={handleSample}
        onRunPipeline={handleRunPipeline}
        onRunGan={() => {
          setTab('gan')
          handleRunGan()
        }}
        busy={busy}
        hasResult={!!result}
        hasGanResult={!!gan}
      />

      <main className="main">
        <header className="topbar">
          <div className="brand">
            <h1>Satellite IR Enhancement &amp; Analysis</h1>
            <p>
              Transform infrared satellite imagery into enhanced, super-resolved and
              colorized RGB with land-cover segmentation, object detection and
              generative (Real-ESRGAN) upscaling.
            </p>
          </div>
          <div className="health" title="Backend status">
            <span className={`dot ${health?.status === 'ok' ? 'ok' : health ? 'bad' : 'warn'}`} />
            <span>
              API {health?.status || 'connecting'} · {health?.device || '—'}
            </span>
          </div>
        </header>

        {health?.status === 'down' ? (
          <Note kind="error">
            Cannot reach the backend on <code>/api</code>. Start it with{' '}
            <code>python -m api.server</code> (or run <code>./run.sh</code>).
          </Note>
        ) : null}

        {busy === 'pipeline' ? (
          <div className="card">
            <div className="hint" style={{ marginBottom: 8 }}>
              Running pipeline: enhance → segment → colorize → report…
            </div>
            <Progress value={progress} />
          </div>
        ) : null}

        {error ? <Note kind="error">{error}</Note> : null}
        {notice ? <Note kind="ok">{notice}</Note> : null}

        {!file ? (
          <div className="panel">
            <div className="steps">
              <div className="step"><b>Step 1</b> — Upload an IR or satellite image, or pick a sample from the sidebar.</div>
              <div className="step"><b>Step 2</b> — Run the full pipeline, or jump straight to the GAN Enhancer tab.</div>
              <div className="step"><b>Step 3</b> — Explore the results across the tabs below.</div>
            </div>
            {health ? <ModelStatus health={health} /> : null}
          </div>
        ) : (
          <div ref={resultRef}>
            <div className="tabs">
              {TABS.map((t) => {
                const disabled = needsResult.includes(t.id) && !result
                return (
                  <button
                    key={t.id}
                    className={`tab${tab === t.id ? ' active' : ''}`}
                    onClick={() => setTab(t.id)}
                    disabled={disabled}
                    title={disabled ? 'Run the pipeline first' : ''}
                  >
                    {t.icon} {t.label}
                  </button>
                )
              })}
            </div>

            <div style={{ marginTop: 18 }}>
              {tab === 'gan' ? (
                <TabGan
                  gan={gan}
                  loading={busy === 'gan'}
                  error={ganError}
                  onRun={handleRunGan}
                  hasFile={!!file}
                />
              ) : !result ? (

                <Empty icon="▤" title="No pipeline results yet">
                  Run the full pipeline from the sidebar to populate this tab.
                </Empty>
              ) : tab === 'overview' ? (
                <TabOverview result={result} />
              ) : tab === 'enhance' ? (
                <TabEnhance result={result} />
              ) : tab === 'colorize' ? (
                <TabColorize result={result} />
              ) : tab === 'analysis' ? (
                <TabAnalysis result={result} />
              ) : (
                <TabReport result={result} />
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

function ModelStatus({ health }) {
  const rows = [
    ['IR enhancer', health.models.enhancer],
    ['IR → RGB colorizer', health.models.colorizer],
    ['Land-cover segmenter', health.models.segmenter],
    ['Scene analyzer', health.models.analyzer],
    ['Trained SR model', health.models.sr],
  ]
  return (
    <div className="card">
      <h3 className="section-title">Backend model status</h3>
      <p className="hint" style={{ marginTop: 2 }}>
        Models without trained weights fall back to classical OpenCV / template implementations
        so the dashboard always works out of the box.
      </p>
      <table style={{ marginTop: 10 }}>
        <thead>
          <tr>
            <th>Component</th>
            <th>Status</th>
            <th>Detail</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, info]) => (
            <tr key={name}>
              <td>{name}</td>
              <td>
                <span className={`tag ${info?.available ? 'ok' : 'fb'}`}>
                  {info?.available ? 'weights' : 'fallback'}
                </span>
              </td>
              <td style={{ color: 'var(--text-faint)' }}>{info?.detail}</td>
            </tr>
          ))}
          <tr>
            <td>GAN enhancer (Real-ESRGAN)</td>
            <td><span className="tag ok">weights</span></td>
            <td style={{ color: 'var(--text-faint)' }}>
              {health.gan.name} · scales {health.gan.scales.join('×, ')}×
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}
