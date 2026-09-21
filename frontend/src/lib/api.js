const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
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
}
