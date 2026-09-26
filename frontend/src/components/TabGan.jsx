import { useState } from 'react'
import { fileUrl } from '../api.js'
import { Figure, Metric, Note, Spinner } from './ui.jsx'

const SCALES = [2, 4]
const TILES = [
  { value: 0, label: 'Whole image (fastest)' },
  { value: 256, label: '256 px tiles' },
  { value: 512, label: '512 px tiles' },
]
const DEVICES = [
  { value: 'auto', label: 'Auto' },
  { value: 'cpu', label: 'CPU' },
  { value: 'cuda', label: 'GPU (CUDA)' },
]

/**
 * Real-ESRGAN super-resolution. The weights shipped in GANs_model/weights are
 * loaded server-side and cached, so repeat runs skip the 67 MB load.
 */
export default function TabGan({ gan, loading, error, onRun, hasFile }) {
  const [scale, setScale] = useState(4)
  const [tile, setTile] = useState(0)
  const [device, setDevice] = useState('auto')

  const dim = (s) => (s ? `${s.width}×${s.height}` : '')

  return (
    <div className="panel">
      <Note kind="info">
        <b>Real-ESRGAN (RRDBNet)</b> — a generative super-resolution GAN. Unlike a plain CNN
        trained with L2 loss, it was trained adversarially, so it hallucinates crisp
        micro-textures instead of producing a smooth blur. All three colour channels pass
        straight through the generator, so output stays in colour.
      </Note>

      <div className="card">
        <h3 className="section-title">Settings</h3>
        <div className="row" style={{ marginTop: 12 }}>
          <div className="field">
            <label htmlFor="gan-scale">Upscale factor</label>
            <select id="gan-scale" value={scale} onChange={(e) => setScale(Number(e.target.value))}>
              {SCALES.map((s) => (
                <option key={s} value={s}>{s}×</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="gan-tile">Tiling</label>
            <select id="gan-tile" value={tile} onChange={(e) => setTile(Number(e.target.value))}>
              {TILES.map((t) => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="gan-device">Device</label>
            <select id="gan-device" value={device} onChange={(e) => setDevice(e.target.value)}>
              {DEVICES.map((d) => (
                <option key={d.value} value={d.value}>{d.label}</option>
              ))}
            </select>
          </div>
        </div>
        <p className="hint" style={{ marginTop: 10 }}>
          Large images on CPU: pick a tile size to bound memory and time. Tiles are blended
          seamlessly across overlaps.
        </p>
        <div className="actions" style={{ marginTop: 12 }}>
          <button
            className="btn primary"
            onClick={() => onRun({ scale, tile, device })}
            disabled={!hasFile || loading}
          >
            {loading ? <><Spinner /> Enhancing…</> : `Enhance ${scale}× with Real-ESRGAN`}
          </button>
        </div>
        {!hasFile ? <p className="hint" style={{ marginTop: 8 }}>Upload an image first.</p> : null}
      </div>

      {error ? <Note kind="error">{error}</Note> : null}

      {gan ? (
        <>
          <div className="metrics">
            <Metric value={`${gan.scale}×`} label="Upscale" />
            <Metric value={dim(gan.shapes.output)} label="Output Size" />
            <Metric value={`${gan.metrics.inference_s}s`} label="Inference Time" />
            <Metric value={`+${gan.metrics.sharpness_gain_pct}%`} label="Sharpness vs Bicubic" />
            <Metric value={gan.device.toUpperCase()} label="Device" />
            <Metric
              value={`${gan.metrics.mean_rgb_in[0]}, ${gan.metrics.mean_rgb_in[1]}, ${gan.metrics.mean_rgb_in[2]}`}
              label="Mean RGB in"
            />
          </div>

          <Figure
            src={fileUrl(gan.urls.comparison)}
            caption="Bicubic baseline (left) vs Real-ESRGAN (right)"
            dim={dim(gan.shapes.output)}
            href
          />

          <div className="grid c3">
            <Figure src={fileUrl(gan.urls.input)} caption="Input" dim={dim(gan.shapes.input)} href />
            <Figure src={fileUrl(gan.urls.bicubic)} caption={`Bicubic ${gan.scale}×`} href />
            <Figure src={fileUrl(gan.urls.output)} caption={`Real-ESRGAN ${gan.scale}×`} dim={dim(gan.shapes.output)} href />
          </div>

          <div className="card">
            <h3 className="section-title">Colour fidelity</h3>
            <p className="hint" style={{ marginTop: 2 }}>
              Mean red/green/blue before and after the GAN — the values stay close, confirming
              the generator is not collapsing the image to greyscale.
            </p>
            <table style={{ marginTop: 10 }}>
              <thead>
                <tr>
                  <th>Stage</th>
                  <th className="num">R</th>
                  <th className="num">G</th>
                  <th className="num">B</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Input</td>
                  {gan.metrics.mean_rgb_in.map((v, i) => <td className="num" key={i}>{v}</td>)}
                </tr>
                <tr>
                  <td>Real-ESRGAN output</td>
                  {gan.metrics.mean_rgb_out.map((v, i) => <td className="num" key={i}>{v}</td>)}
                </tr>
              </tbody>
            </table>
          </div>

          <div className="actions">
            <a className="btn primary" href={fileUrl(gan.urls.output)} download={`realesrgan_x${gan.scale}.png`}>
              Download Enhanced PNG
            </a>
            <a className="btn" href={fileUrl(gan.urls.comparison)} download={`realesrgan_x${gan.scale}_comparison.png`}>
              Download Comparison
            </a>
          </div>
        </>
      ) : (
        <div className="empty">
          <div className="ico">◈</div>
          <div className="t">No GAN result yet</div>
          <div className="s">
            Pick an upscale factor and run the enhancer to compare Real-ESRGAN against a
            bicubic baseline on your image.
          </div>
        </div>
      )}
    </div>
  )
}
