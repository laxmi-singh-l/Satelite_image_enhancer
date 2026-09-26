import { useRef, useState } from 'react'
import { fileUrl } from '../api.js'

export default function Sidebar({
  file,
  previewUrl,
  onFile,
  samples,
  selectedSample,
  onSelectSample,
  onRunPipeline,
  onRunGan,
  busy,
  hasResult,
  hasGanResult,
}) {
  const inputRef = useRef(null)
  const [drag, setDrag] = useState(false)

  const pick = (f) => {
    if (!f) return
    if (!f.type.startsWith('image/') && !/\.(png|jpe?g|tiff?)$/i.test(f.name)) return
    onFile(f)
    onSelectSample(null)
  }

  const grouped = samples.reduce((acc, s) => {
    ;(acc[s.group] ||= []).push(s)
    return acc
  }, {})

  return (
    <aside className="sidebar">
      <div>
        <div className="side-title">Input Image</div>
        <div
          className={`dropzone${drag ? ' drag' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDrag(true)
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDrag(false)
            pick(e.dataTransfer.files?.[0])
          }}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
        >
          <div className="big">↑</div>
          <div className="t">Upload an image</div>
          <div className="s">click or drag &amp; drop · png / jpg / tif</div>
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg,image/tiff,.png,.jpg,.jpeg,.tif,.tiff"
            onChange={(e) => pick(e.target.files?.[0])}
          />
        </div>

        {file ? (
          <>
            <div className="file-chip">{file.name}</div>
            <div className="preview" style={{ marginTop: 10 }}>
              <img src={previewUrl} alt="Selected input preview" />
              <div className="cap">
                Local preview · colour preserved
              </div>
            </div>
          </>
        ) : null}
      </div>

      <div>
        <div className="actions" style={{ flexDirection: 'column' }}>
          <button className="btn primary block" onClick={onRunPipeline} disabled={!file || busy}>
            {busy === 'pipeline' ? <><span className="spinner" /> Processing…</> : 'Run Full Pipeline'}
          </button>
          <button className="btn block" onClick={onRunGan} disabled={!file || busy}>
            {busy === 'gan' ? <><span className="spinner" /> Enhancing…</> : 'GAN Enhance (Real-ESRGAN)'}
          </button>
        </div>
        {!file ? <p className="hint" style={{ marginTop: 8 }}>Upload an image to enable processing.</p> : null}
        {file && !hasResult ? <p className="hint" style={{ marginTop: 8 }}>Run the pipeline to populate the analysis tabs.</p> : null}
        {file && hasResult && !hasGanResult ? <p className="hint" style={{ marginTop: 8 }}>Tip: try the GAN Enhancer tab for 2x/4x upscaling.</p> : null}
      </div>

      <div>
        <div className="side-title">Sample Images</div>
        {!samples.length ? (
          <p className="hint">Loading samples…</p>
        ) : (
          Object.entries(grouped).map(([group, items]) => (
            <div key={group} style={{ marginBottom: 12 }}>
              <p className="hint" style={{ margin: '0 0 5px' }}>{group}</p>
              <div className="sample-grid">
                {items.slice(0, 9).map((s) => (
                  <button
                    key={`${s.group}/${s.name}`}
                    className={selectedSample === s.name ? 'sel' : ''}
                    title={s.name}
                    onClick={() => onSelectSample(s)}
                  >
                    <img src={fileUrl(s.url)} alt={s.name} loading="lazy" />
                  </button>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </aside>
  )
}
