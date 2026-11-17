import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import RouteMetrics from './RouteMetrics'

function precision(hits, categoria) {
  if (!categoria || !hits?.length) return null
  return hits.filter((h) => h.categoria === categoria).length / hits.length
}

function Thumbs({ hits, categoria, onPick }) {
  if (!hits?.length) return <div className="empty"><p>Sin resultados.</p></div>
  return (
    <div className="thumbs">
      {hits.map((h, i) => {
        const ok = categoria ? h.categoria === categoria : null
        return (
          <figure key={h.id} className={ok == null ? '' : ok ? 'ok' : 'miss'} onClick={() => onPick?.(h)}>
            <img src={api.imageUrl(h.id)} alt={h.categoria} loading="lazy" />
            <figcaption>
              <span className="rank">{i + 1}</span>
              <span className="cat">{h.categoria}</span>
              <span className="num">{h.score.toFixed(3)}</span>
            </figcaption>
          </figure>
        )
      })}
    </div>
  )
}

export default function MultimediaPanel() {
  const [status, setStatus] = useState(null)
  const [samples, setSamples] = useState([])
  const [query, setQuery] = useState(null) // {kind:'indexed', id, categoria} | {kind:'upload', file, url}
  const [k, setK] = useState(10)
  const [probes, setProbes] = useState(5)
  const [kWords, setKWords] = useState(256)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [res, setRes] = useState(null)
  const timer = useRef(null)

  const refrescar = useCallback(async () => {
    try {
      const s = await api.mmStatus()
      setStatus(s)
      return s
    } catch (e) {
      setError(e)
    }
  }, [])

  const cargarMuestras = useCallback(async () => {
    try {
      const r = await api.mmSamples(12, Math.floor(Math.random() * 1e6))
      setSamples(r.samples)
    } catch {
      setSamples([])
    }
  }, [])

  useEffect(() => {
    refrescar().then((s) => s?.ready && cargarMuestras())
    return () => clearInterval(timer.current)
  }, [refrescar, cargarMuestras])

  useEffect(() => {
    clearInterval(timer.current)
    if (status?.building) {
      timer.current = setInterval(async () => {
        const s = await refrescar()
        if (s && !s.building && s.ready) cargarMuestras()
      }, 2000)
    }
    return () => clearInterval(timer.current)
  }, [status?.building, refrescar, cargarMuestras])

  async function construir() {
    setError(null)
    try {
      await api.mmBuild({ k_words: kWords })
      await refrescar()
    } catch (e) {
      setError(e)
    }
  }

  async function buscar(q = query) {
    if (!q || busy) return
    setBusy(true)
    setError(null)
    try {
      const r = await api.mmSearch(
        q.kind === 'upload' ? { file: q.file, k, probes } : { imageId: q.id, k, probes },
      )
      setRes(r)
    } catch (e) {
      setError(e)
      setRes(null)
    } finally {
      setBusy(false)
    }
  }

  function elegirMuestra(s) {
    const q = { kind: 'indexed', id: s.id, categoria: s.categoria }
    setQuery(q)
    buscar(q)
  }

  function subir(e) {
    const file = e.target.files?.[0]
    if (!file) return
    const q = { kind: 'upload', file, url: URL.createObjectURL(file), nombre: file.name }
    setQuery(q)
    buscar(q)
  }

  const listo = status?.ready
  const cat = query?.kind === 'indexed' ? query.categoria : null
  const pB = res ? precision(res.bovw.hits, cat) : null
  const pD = res ? precision(res.dense.hits, cat) : null

  return (
    <div className="ia">
      <section className="ia-status">
        <div>
          <h2>Búsqueda de imágenes: clásico vs. moderno</h2>
          <p className="muted">
            Bag of Visual Words (SIFT 128-D → K-Means → índice invertido con TF-IDF y coseno) frente a
            embeddings densos (ResNet-18 512-D → IVF en disco), sobre Caltech-101.
          </p>
        </div>
        <dl className="facts-grid">
          <div><dt>imágenes</dt><dd>{listo ? status.n_images.toLocaleString('es') : '—'}</dd></div>
          <div><dt>palabras visuales</dt><dd>{listo ? status.k_words : '—'}</dd></div>
          <div><dt>postings BoVW</dt><dd>{listo ? status.bovw.postings.toLocaleString('es') : '—'}</dd></div>
          <div><dt>páginas postings</dt><dd>{listo ? status.bovw.postings_pages : '—'}</dd></div>
          <div><dt>celdas IVF</dt><dd>{listo ? status.dense_index.n_lists : '—'}</dd></div>
          <div><dt>páginas tabla</dt><dd>{listo ? status.table_pages : '—'}</dd></div>
        </dl>
        <div className="row-actions">
          <label className="inline">
            K
            <select value={kWords} onChange={(e) => setKWords(Number(e.target.value))}>
              {[128, 256, 512].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <button className="action" onClick={construir} disabled={status?.building}>
            {listo ? 'Reconstruir tienda' : 'Construir tienda'}
          </button>
          {status?.building && (
            <span className="progress">
              {status.stage}
              {status.total ? ` ${status.done}/${status.total}` : ''}
            </span>
          )}
          {status?.error && <span className="progress error">{status.error}</span>}
        </div>
      </section>

      <section className="ia-query mm">
        <div className="mm-query">
          <div className="query-image">
            {query ? (
              <img src={query.kind === 'upload' ? query.url : api.imageUrl(query.id)} alt="consulta" />
            ) : (
              <div className="placeholder">imagen de consulta</div>
            )}
            <span className="muted small-text">
              {query ? (query.kind === 'upload' ? query.nombre : `${query.categoria} · id ${query.id}`) : ''}
            </span>
          </div>
          <div className="mm-side">
            <div className="ia-controls">
              <label className="file">
                <input type="file" accept="image/*" onChange={subir} disabled={!listo} />
                Subir una foto
              </label>
              <label>k <input type="number" min="1" max="50" value={k} onChange={(e) => setK(Number(e.target.value))} /></label>
              <label>probes <input type="number" min="1" max="200" value={probes} onChange={(e) => setProbes(Number(e.target.value))} /></label>
              <button className="action primary" onClick={() => buscar()} disabled={busy || !query || !listo}>
                {busy ? 'Buscando' : 'Buscar'}
              </button>
            </div>
            <div className="samples-head">
              <span className="muted">o elige una imagen indexada como consulta</span>
              <button className="action" onClick={cargarMuestras} disabled={!listo}>otra muestra</button>
            </div>
            <div className="thumbs small">
              {samples.map((s) => (
                <figure key={s.id} className={query?.id === s.id ? 'sel' : ''} onClick={() => elegirMuestra(s)}>
                  <img src={api.imageUrl(s.id)} alt={s.categoria} loading="lazy" />
                  <figcaption><span className="cat">{s.categoria}</span></figcaption>
                </figure>
              ))}
            </div>
          </div>
        </div>
      </section>

      <div className="scroll ia-body">
        {error && (
          <div className="notice error">
            <h3>No se pudo completar</h3>
            <p>{error.message}</p>
          </div>
        )}
        {!listo && !error && (
          <div className="notice pending">
            <h3>La tienda de imágenes todavía no está construida</h3>
            <p>
              Pulsa <b>Construir tienda</b>: extrae SIFT y ResNet-18 de las 8 677 imágenes de
              Caltech-101, entrena el codebook, carga la tabla <code>imagenes</code> y construye el
              IVF y el índice invertido. Tarda varios minutos en CPU.
            </p>
          </div>
        )}
        {res && (
          <div className="cols2">
            <section className="route">
              <header>
                <h3>Clásico · BoVW</h3>
                <span className="muted">SIFT → {status?.k_words} palabras → índice invertido (TF-IDF · coseno)</span>
              </header>
              <RouteMetrics
                metrics={res.bovw.metrics}
                extra={[
                  ['extracción SIFT + cuantización', `${res.bovw.extract_ms} ms`],
                  ['descriptores', res.bovw.n_descriptors],
                  ...(pB != null ? [[`Precision@${res.bovw.hits.length}`, `${(100 * pB).toFixed(0)} %`]] : []),
                ]}
              />
              <Thumbs hits={res.bovw.hits} categoria={cat} onPick={elegirMuestra} />
            </section>
            <section className="route">
              <header>
                <h3>Moderno · ResNet-18</h3>
                <span className="muted">
                  <code>ORDER BY emb &lt;=&gt; '[…]' LIMIT {res.k}</code> · IVF COSINE
                </span>
              </header>
              <RouteMetrics
                metrics={res.dense.metrics}
                extra={[
                  ['extracción ResNet-18', `${res.dense.extract_ms} ms`],
                  ...(pD != null ? [[`Precision@${res.dense.hits.length}`, `${(100 * pD).toFixed(0)} %`]] : []),
                ]}
              />
              <Thumbs hits={res.dense.hits} categoria={cat} onPick={elegirMuestra} />
            </section>
          </div>
        )}
        {res && cat && (
          <p className="muted legend-note">
            Borde verde: misma categoría que la consulta («{cat}»); rojo: categoría distinta. La
            propia imagen de consulta se excluye del ranking.
          </p>
        )}
      </div>
    </div>
  )
}
