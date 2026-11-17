const BASE = '/api'

async function request(path, options = {}) {
  const esForm = options.body instanceof FormData
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    // con FormData el navegador fija el Content-Type (incluye el boundary)
    headers: esForm ? options.headers : { 'Content-Type': 'application/json', ...(options.headers ?? {}) },
  })
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    // El backend envia {detail: {error, detail, missing?}}; se normaliza aqui
    // para que la interfaz distinga "esto esta mal" de "esto falta por construir".
    const d = body?.detail ?? {}
    throw Object.assign(new Error(d.detail ?? res.statusText), {
      kind: d.error ?? 'desconocido',
      missing: d.missing,
      status: res.status,
    })
  }
  return body
}

export const api = {
  health: () => request('/health'),
  query: (sql, maxRows = 500) =>
    request('/query', { method: 'POST', body: JSON.stringify({ sql, max_rows: maxRows }) }),
  tables: () => request('/tables'),
  pages: (table, offset = 0, limit = 120) =>
    request(`/tables/${encodeURIComponent(table)}/pages?offset=${offset}&limit=${limit}`),
  reorganize: (table) =>
    request(`/tables/${encodeURIComponent(table)}/reorganize`, { method: 'POST' }),
  seed: (table, body) =>
    request(`/tables/${encodeURIComponent(table)}/seed`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  // ---- aplicacion de IA (RAG) y modulo multimedia (Rol 4)
  ragStatus: () => request('/rag/status'),
  ragBuild: (body = {}) => request('/rag/build', { method: 'POST', body: JSON.stringify(body) }),
  ragAsk: (body) => request('/rag/ask', { method: 'POST', body: JSON.stringify(body) }),
  mmStatus: () => request('/multimedia/status'),
  mmBuild: (body = {}) => request('/multimedia/build', { method: 'POST', body: JSON.stringify(body) }),
  mmSamples: (n = 12, seed) =>
    request(`/multimedia/samples?n=${n}${seed != null ? `&seed=${seed}` : ''}`),
  mmSearch: ({ file, imageId, k = 10, probes = 5 }) => {
    const form = new FormData()
    if (file) form.append('file', file)
    if (imageId != null) form.append('image_id', String(imageId))
    form.append('k', String(k))
    form.append('probes', String(probes))
    return request('/multimedia/search', { method: 'POST', body: form })
  },
  imageUrl: (id) => `${BASE}/multimedia/image/${id}`,
}
