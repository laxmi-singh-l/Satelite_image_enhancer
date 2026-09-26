import { BarChart, Figure, Legend, Metric, Note, CLASS_COLORS } from './ui.jsx'
import { fileUrl, downloadUrl } from '../api.js'

const dim = (s) => (s ? `${s.width}×${s.height} · ${s.channels}ch` : '')
const clsEntries = Object.entries(CLASS_COLORS)

export function TabOverview({ result }) {
  const r = result.report
  const stats = r?.summary_stats || {}
  return (
    <div className="panel">
      <div className="metrics">
        <Metric
          value={stats.dominant_class ? stats.dominant_class.charAt(0).toUpperCase() + stats.dominant_class.slice(1) : '—'}
          label="Dominant Class"
        />
        <Metric value={`${(stats.natural_coverage ?? 0).toFixed(1)}%`} label="Natural Coverage" />
        <Metric value={`${(stats.anthropogenic_coverage ?? 0).toFixed(1)}%`} label="Anthropogenic" />
        <Metric value={result.objects.length} label="Objects Detected" />
        <Metric value={`${Math.round((r?.confidence ?? 0) * 100)}%`} label="Confidence" />
        <Metric value={`${result.timings.total_s}s`} label="Total Time" />
      </div>

      <div className="card">
        <h3 className="section-title">Scene Description</h3>
        <p className="hint" style={{ marginTop: 2 }}>Natural-language summary of the analysed scene.</p>
        <div className="report-box" style={{ marginTop: 10 }}>
          {r?.description || 'No description available.'}
        </div>
      </div>

      <div className="card">
        <h3 className="section-title">Land Cover Composition</h3>
        <p className="hint" style={{ marginTop: 2 }}>Share of the scene per classified class.</p>
        <div style={{ marginTop: 12 }}>
          <BarChart data={Object.fromEntries(
            Object.entries(result.land_cover || {}).map(([k, v]) => [k, v.percentage]),
          )} />
        </div>
        <div style={{ marginTop: 14 }}>
          <Legend classes={clsEntries} />
        </div>
      </div>

      <div className="grid c2">
        <Figure src={fileUrl(result.urls.input)} caption="Input" dim={dim(result.shapes.input)} href />
        <Figure src={fileUrl(result.urls.colorized)} caption="Colorized RGB" dim={dim(result.shapes.colorized)} href />
      </div>
    </div>
  )
}

export function TabEnhance({ result }) {
  return (
    <div className="panel">
      <p className="sub">
        Infrared enhancement with CLAHE + denoising, then {result.scale}× upscaling.
        Output dimensions {dim(result.shapes.enhanced)}.
      </p>
      <Figure src={fileUrl(result.urls.comparison)} caption="Before (left) vs after (right) enhancement" href />
      <div className="grid c2">
        <Figure src={fileUrl(result.urls.input)} caption="Original" dim={dim(result.shapes.input)} href />
        <Figure src={fileUrl(result.urls.enhanced)} caption="Enhanced" dim={dim(result.shapes.enhanced)} href />
      </div>
      <div className="card">
        <h3 className="section-title">Stage timings</h3>
        <div className="row" style={{ marginTop: 10 }}>
          {Object.entries(result.timings).map(([k, v]) => (
            <Metric key={k} value={`${v}s`} label={k.replace(/_s$/, '').replace(/_/g, ' ')} />
          ))}
        </div>
      </div>
    </div>
  )
}

export function TabColorize({ result }) {
  return (
    <div className="panel">
      <p className="sub">Single-band infrared rasters rendered to true-colour RGB, guided by the land-cover mask.</p>
      <div className="grid c3">
        <Figure src={fileUrl(result.urls.enhanced)} caption="Enhanced IR (input)" dim={dim(result.shapes.enhanced)} href />
        <Figure src={fileUrl(result.urls.colorized)} caption="Colorized RGB (output)" dim={dim(result.shapes.colorized)} href />
        <Figure src={fileUrl(result.urls.overlay)} caption="Segmentation overlay" href />
      </div>
      <Note kind="info">
        Images are served as full-colour PNGs straight from the API — no greyscale conversion is
        applied on upload or on download.
      </Note>
    </div>
  )
}

export function TabAnalysis({ result }) {
  const objects = result.objects || []
  const lc = Object.entries(result.land_cover || {}).sort((a, b) => b[1].percentage - a[1].percentage)
  return (
    <div className="panel">
      <div className="grid c2">
        <Figure src={fileUrl(result.urls.segmentation)} caption="Segmentation map" href />
        <Figure src={fileUrl(result.urls.overlay)} caption="Overlay on colorized RGB" href />
      </div>

      <div className="card">
        <h3 className="section-title">Legend</h3>
        <div style={{ marginTop: 10 }}>
          <Legend classes={clsEntries} />
        </div>
      </div>

      {result.urls.detections ? (
        <Figure src={fileUrl(result.urls.detections)} caption={`${objects.length} objects detected`} href />
      ) : null}

      <div className="card">
        <h3 className="section-title">Detected Objects</h3>
        {objects.length ? (
          <table>
            <thead>
              <tr>
                <th>Class</th>
                <th>Confidence</th>
                <th className="num">BBox (x1, y1, x2, y2)</th>
              </tr>
            </thead>
            <tbody>
              {objects.map((o, i) => (
                <tr key={i}>
                  <td>{o.class}</td>
                  <td className="num">{Number(o.confidence).toFixed(2)}</td>
                  <td className="num" style={{ color: 'var(--text-faint)' }}>
                    {o.bbox.map((v) => Math.round(v)).join(', ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="hint" style={{ marginTop: 8 }}>No objects detected in this scene.</p>
        )}
      </div>

      <div className="card">
        <h3 className="section-title">Land Cover Statistics</h3>
        {lc.length ? (
          <table style={{ marginTop: 8 }}>
            <thead>
              <tr>
                <th>Class</th>
                <th className="num">Coverage (%)</th>
                <th className="num">Pixels</th>
              </tr>
            </thead>
            <tbody>
              {lc.map(([name, s]) => (
                <tr key={name}>
                  <td>
                    <span className="swatch" style={{ background: CLASS_COLORS[name] || '#888' }} />
                    {name.charAt(0).toUpperCase() + name.slice(1)}
                  </td>
                  <td className="num">{s.percentage}</td>
                  <td className="num">{s.pixel_count.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="hint" style={{ marginTop: 8 }}>No land-cover statistics.</p>
        )}
      </div>
    </div>
  )
}

export function TabReport({ result }) {
  const r = result.report
  return (
    <div className="panel">
      <div className="grid c2">
        <div className="card">
          <h3 className="section-title">Scene Metadata</h3>
          <dl className="kv" style={{ marginTop: 10 }}>
            <dt>Scene type</dt>
            <dd>{r?.scene_type ?? '—'}</dd>
            <dt>Confidence</dt>
            <dd>{Math.round((r?.confidence ?? 0) * 100)}%</dd>
            <dt>Analysis time</dt>
            <dd>{r?.time_analysis ?? '—'}</dd>
            <dt>Classes detected</dt>
            <dd>{Object.keys(r?.land_cover || {}).length}</dd>
          </dl>
        </div>
        <div className="card">
          <h3 className="section-title">Object Detection Summary</h3>
          {r?.object_counts && Object.keys(r.object_counts).length ? (
            <dl className="kv" style={{ marginTop: 10 }}>
              {Object.entries(r.object_counts).map(([k, v]) => (
                <div key={k} style={{ display: 'contents' }}>
                  <dt>{k}</dt>
                  <dd>{v}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="hint" style={{ marginTop: 8 }}>No objects detected.</p>
          )}
        </div>
      </div>

      <div className="card">
        <h3 className="section-title">Natural Language Description</h3>
        <div className="report-box" style={{ marginTop: 10 }}>{r?.description || '—'}</div>
      </div>

      <div className="card">
        <h3 className="section-title">Raw Report (JSON)</h3>
        <pre className="json" style={{ marginTop: 10 }}>{JSON.stringify(r, null, 2)}</pre>
      </div>

      <div className="actions">
        <a className="btn" href={fileUrl(result.urls.colorized)} download="colorized_rgb.png">
          Download Colorized RGB (PNG)
        </a>
        <a className="btn" href={fileUrl(result.urls.enhanced)} download="enhanced_ir.png">
          Download Enhanced IR (PNG)
        </a>
        <a className="btn" href={fileUrl(result.urls.segmentation)} download="segmentation.png">
          Download Segmentation (PNG)
        </a>
        <a className="btn" href={downloadUrl(result.job)}>Download All (ZIP)</a>
      </div>
    </div>
  )
}
