export function Note({ kind = 'info', children }) {
  const icons = { info: 'i', warn: '!', error: 'x', ok: '+' }
  return (
    <div className={`note ${kind}`}>
      <span className="ico">{icons[kind]}</span>
      <div>{children}</div>
    </div>
  )
}

export function Metric({ value, label }) {
  return (
    <div className="metric">
      <div className="v">{value ?? '—'}</div>
      <div className="l">{label}</div>
    </div>
  )
}

/**
 * Image panel. `dim` renders the pixel size next to the caption so it is
 * obvious the image is being served in full colour, unmodified.
 */
export function Figure({ src, caption, dim, href }) {
  if (!src) return null
  const img = <img src={src} alt={caption} />
  return (
    <figure>
      {href ? (
        <a href={src} target="_blank" rel="noreferrer" style={{ display: 'block' }}>
          {img}
        </a>
      ) : (
        img
      )}
      <figcaption>
        <span>{caption}</span>
        {dim ? <span className="dim">{dim}</span> : null}
      </figcaption>
    </figure>
  )
}

export function BarChart({ data }) {
  const rows = Object.entries(data || {}).sort((a, b) => b[1] - a[1])
  if (!rows.length) return <p className="hint">No land-cover data.</p>
  const max = Math.max(...rows.map(([, v]) => v), 1)
  return (
    <div className="bars">
      {rows.map(([name, value]) => (
        <div className="bar-row" key={name}>
          <span style={{ textTransform: 'capitalize' }}>{name}</span>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{ width: `${(value / max) * 100}%`, background: CLASS_COLORS[name] || '#4aa8ff' }}
            />
          </div>
          <span className="bar-val">{Number(value).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

export const CLASS_COLORS = {
  water: 'rgb(60,119,181)',
  forest: 'rgb(34,139,34)',
  agriculture: 'rgb(154,205,50)',
  urban: 'rgb(178,34,34)',
  barren: 'rgb(210,180,140)',
  wetland: 'rgb(0,206,209)',
  grassland: 'rgb(124,252,0)',
}

export function Legend({ classes }) {
  return (
    <div className="legend">
      {classes.map(([name, color]) => (
        <span key={name}>
          <span className="swatch" style={{ background: `rgb(${color})` }} />
          {name.charAt(0).toUpperCase() + name.slice(1)}
        </span>
      ))}
    </div>
  )
}

export function Progress({ value }) {
  return (
    <div className="progress">
      <i style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  )
}

export function Spinner() {
  return <span className="spinner" />
}

export function Empty({ icon = '·', title, children }) {
  return (
    <div className="empty">
      <div className="ico">{icon}</div>
      <div className="t">{title}</div>
      <div className="s">{children}</div>
    </div>
  )
}
