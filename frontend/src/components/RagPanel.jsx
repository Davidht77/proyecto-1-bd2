import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import RouteMetrics from './RouteMetrics'

const EJEMPLOS = [
  '¿Qué es un agujero negro y cómo se forma?',
  '¿Cuál es la capital de Andorra y cuántos habitantes tiene?',
  'What is the theory of general relativity?',
  '¿Quién fue Ada Lovelace?',
  'How does photosynthesis work?',
]

const MODOS = [
  ['semantic', 'semántico (SBERT + IVF)'],
  ['lexical', 'léxico (índice invertido)'],
  ['both', 'ambos (fusión por rango)'],
]

function bytes(n) {
  if (n == null) return '—'
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

/** Pinta las citas [n] de la respuesta como chips enlazadas a la fuente. */
function Respuesta({ texto }) {
  // Acepta [1], [1][2] y la forma agrupada [1, 2] que a veces escribe el modelo.
  const partes = texto.split(/(\[\d+(?:\s*,\s*\d+)*\])/g)
  return (
    <p className="answer-text">
      {partes.map((p, i) => {
        const m = p.match(/^\[([\d,\s]+)\]$/)
        if (!m) return <span key={i}>{p}</span>
        return m[1]
          .split(',')
          .map((n) => n.trim())
          .map((n) => (
            <a key={`${i}-${n}`} className="cite" href={`#fuente-${n}`}>
              {n}
            </a>
          ))
      })}
    </p>
  )
}

function Hits({ hits, resaltar }) {
  if (!hits?.length) return <div className="empty"><p>Sin resultados.</p></div>
  return (
    <ol className="hits">
      {hits.map((h, i) => (
        <li key={h.id} className={resaltar?.has(h.id) ? 'hit both' : 'hit'}>
          <div className="hit-head">
            <span className="rank">{i + 1}</span>
            <span className="title">{h.titulo}</span>
            <span className="chip">{h.lang}</span>
            <span className="num">{typeof h.score === 'number' ? h.score.toFixed(4) : h.score}</span>
          </div>
          <p className="snippet">{h.texto.slice(0, 220)}{h.texto.length > 220 ? '…' : ''}</p>
        </li>
      ))}
    </ol>
  )
}

export default function RagPanel() {
  const [status, setStatus] = useState(null)
  const [pregunta, setPregunta] = useState(EJEMPLOS[0])
  const [k, setK] = useState(5)
  const [modo, setModo] = useState('semantic')
  const [probes, setProbes] = useState(5)
  const [scorer, setScorer] = useState('tfidf')
  const [generar, setGenerar] = useState(true)
  const [limit, setLimit] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [res, setRes] = useState(null)
  const timer = useRef(null)

  const refrescar = useCallback(async () => {
    try {
      const s = await api.ragStatus()
      setStatus(s)
      return s
    } catch (e) {
      setError(e)
    }
  }, [])

  useEffect(() => {
    refrescar()
    return () => clearInterval(timer.current)
  }, [refrescar])

  // Mientras se construye el corpus, se consulta el progreso cada 2 s.
  useEffect(() => {
    clearInterval(timer.current)
    if (status?.building) timer.current = setInterval(refrescar, 2000)
    return () => clearInterval(timer.current)
  }, [status?.building, refrescar])

  async function construir() {
    setError(null)
    try {
      await api.ragBuild(limit ? { limit: Number(limit) } : {})
      await refrescar()
    } catch (e) {
      setError(e)
    }
  }

  async function preguntar() {
    if (!pregunta.trim() || busy) return
    setBusy(true)
    setError(null)
    try {
      setRes(await api.ragAsk({ question: pregunta, k, mode: modo, probes, scorer, generate: generar }))
    } catch (e) {
      setError(e)
      setRes(null)
    } finally {
      setBusy(false)
    }
  }

  const listo = status?.ready
  const comunes = res
    ? new Set(res.lexical.hits.map((h) => h.id).filter((id) => res.semantic.hits.some((s) => s.id === id)))
    : null

  return (
    <div className="ia">
      <section className="ia-status">
        <div>
          <h2>RAG sobre Wikipedia</h2>
          <p className="muted">
            Recuperación léxica (índice invertido en disco, TF-IDF/BM25) frente a recuperación
            semántica (SBERT + IVF en disco); el modelo de lenguaje responde citando los fragmentos.
          </p>
        </div>
        <dl className="facts-grid">
          <div><dt>documentos</dt><dd>{listo ? status.n_docs.toLocaleString('es') : '—'}</dd></div>
          <div><dt>páginas de la tabla</dt><dd>{listo ? status.table_pages.toLocaleString('es') : '—'}</dd></div>
          <div><dt>celdas IVF</dt><dd>{listo ? status.dense_index.n_lists : '—'}</dd></div>
          <div><dt>vocabulario</dt><dd>{listo ? bytes(status.inverted_bytes?.['vocab.dic']) : '—'}</dd></div>
          <div><dt>postings</dt><dd>{listo ? bytes(status.inverted_bytes?.['postings.inv']) : '—'}</dd></div>
          <div>
            <dt>LLM</dt>
            <dd title={status?.llm?.model}>{status?.llm ? `${status.llm.provider} · ${status.llm.model}` : '—'}</dd>
          </div>
        </dl>
        <div className="row-actions">
          <input
            className="small"
            type="number"
            min="10"
            placeholder="tope de docs (vacío = todos)"
            value={limit}
            onChange={(e) => setLimit(e.target.value)}
          />
          <button className="action" onClick={construir} disabled={status?.building}>
            {listo ? 'Reconstruir corpus' : 'Construir corpus'}
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

      <section className="ia-query">
        <textarea
          value={pregunta}
          onChange={(e) => setPregunta(e.target.value)}
          onKeyDown={(e) => (e.ctrlKey || e.metaKey) && e.key === 'Enter' && preguntar()}
          placeholder="Escribe una pregunta en español o inglés"
          aria-label="Pregunta"
        />
        <div className="samples">
          {EJEMPLOS.map((q) => (
            <button key={q} onClick={() => setPregunta(q)}>{q}</button>
          ))}
        </div>
        <div className="ia-controls">
          <label>k <input type="number" min="1" max="20" value={k} onChange={(e) => setK(Number(e.target.value))} /></label>
          <label>
            contexto
            <select value={modo} onChange={(e) => setModo(e.target.value)}>
              {MODOS.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
            </select>
          </label>
          <label>probes <input type="number" min="1" max="200" value={probes} onChange={(e) => setProbes(Number(e.target.value))} /></label>
          <label>
            léxico
            <select value={scorer} onChange={(e) => setScorer(e.target.value)}>
              <option value="tfidf">TF-IDF · coseno</option>
              <option value="bm25">BM25</option>
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={generar} onChange={(e) => setGenerar(e.target.checked)} /> generar respuesta
          </label>
          <span className="spacer" />
          <button className="action primary" onClick={preguntar} disabled={busy || !listo}>
            {busy ? 'Consultando' : 'Preguntar'}
          </button>
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
            <h3>El corpus todavía no está cargado</h3>
            <p>
              Pulsa <b>Construir corpus</b>: calcula los embeddings SBERT de los artículos, los carga
              en la tabla <code>documentos</code>, construye el IVF y el índice invertido en disco.
              Con 15 000 artículos tarda varios minutos en CPU.
            </p>
          </div>
        )}

        {res?.answer && (
          <section className="answer">
            <header>
              <span className="chip strong">{res.answer.provider}</span>
              <span className="muted">{res.answer.model}</span>
              <span className="spacer" />
              <span className="num">{res.answer.ms.toLocaleString('es')} ms</span>
              {res.answer.output_tokens != null && (
                <span className="num muted">{res.answer.input_tokens}→{res.answer.output_tokens} tok</span>
              )}
            </header>
            <Respuesta texto={res.answer.text} />
            <ol className="sources">
              {res.context.map((c) => (
                <li key={c.n} id={`fuente-${c.n}`}>
                  <span className="rank">{c.n}</span> <b>{c.titulo}</b> <span className="chip">{c.lang}</span>
                  <span className="muted"> · id {c.id}</span>
                </li>
              ))}
            </ol>
          </section>
        )}

        {res && (
          <div className="cols2">
            <section className="route">
              <header>
                <h3>Léxico · {res.lexical.scorer === 'bm25' ? 'BM25' : 'TF-IDF + coseno'}</h3>
                <span className="muted">índice invertido en disco (vocab B+ + postings)</span>
              </header>
              <RouteMetrics metrics={res.lexical.metrics} />
              <Hits hits={res.lexical.hits} resaltar={comunes} />
            </section>
            <section className="route">
              <header>
                <h3>Semántico · SBERT 384-D</h3>
                <span className="muted">
                  <code>ORDER BY emb &lt;=&gt; '[…]' LIMIT {res.k}</code> resuelto por el planificador
                </span>
              </header>
              <RouteMetrics metrics={res.semantic.metrics} />
              <Hits hits={res.semantic.hits} resaltar={comunes} />
            </section>
          </div>
        )}
        {res && comunes && (
          <p className="muted legend-note">
            Los fragmentos resaltados aparecen en las dos rutas ({comunes.size} de {res.k}).
          </p>
        )}
      </div>
    </div>
  )
}
