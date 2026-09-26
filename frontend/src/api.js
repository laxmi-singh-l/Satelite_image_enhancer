const BASE = ''

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return res.json()
}

export function getHealth() {
  return request('/api/health')
}

export function getSamples() {
  return request('/api/samples')
}

export function runPipeline(file) {
  const fd = new FormData()
  fd.append('file', file)
  return request('/api/analyze', { method: 'POST', body: fd })
}

export function runGan(file, { scale = 4, tile = 0, device = 'auto' } = {}) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('scale', String(scale))
  fd.append('tile', String(tile))
  fd.append('device', device)
  return request('/api/gan-enhance', { method: 'POST', body: fd })
}

export function downloadUrl(job) {
  return `/api/download/${job}`
}

export const fileUrl = (u) => (u ? `${BASE}${u}` : null)
