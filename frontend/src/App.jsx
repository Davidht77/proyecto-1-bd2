import { useCallback, useEffect, useState } from 'react'
import { api } from './lib/api'
import TableExplorer from './components/TableExplorer'
import PlanMetrics from './components/PlanMetrics'
import ResultsTable from './components/ResultsTable'
import PageMap from './components/PageMap'
import MapView from './components/MapView'
import RagPanel from './components/RagPanel'
import MultimediaPanel from './components/MultimediaPanel'

const EJEMPLOS = [
  ['crear tabla', "CREATE TABLE empleados (\n  id INT PRIMARY KEY,\n  nombre CHAR(30),\n  dept CHAR(20),\n  salario FLOAT\n) USING SEQUENTIAL"],
  ['insertar', "INSERT INTO empleados VALUES (101, 'Ada Lovelace', 'Analytics', 5200.0)"],
  ['igualdad', 'SELECT * FROM empleados WHERE id = 101'],
  ['rango', 'SELECT * FROM empleados WHERE id >= 100 AND id <= 500'],
  ['escaneo completo', "SELECT * FROM empleados WHERE dept = 'Analytics'"],
  ['borrar', 'DELETE FROM empleados WHERE id = 101'],
  ['crear tabla espacial', "CREATE TABLE locales (\n  id INT PRIMARY KEY,\n  nombre CHAR(20),\n  ubicacion POINT,\n  categoria CHAR(10)\n) USING HEAP"],
  ['insertar punto', "INSERT INTO locales VALUES (1, 'Miraflores', POINT(-77.0282, -12.1354), 'tienda')"],
  ['índice R-Tree', 'CREATE INDEX idx_geo ON locales (ubicacion) USING RTREE'],
  ['radio', 'SELECT id, nombre, ubicacion FROM locales WHERE ubicacion WITHIN_RADIUS ((-77.0282, -12.1354), 2.5)'],
  ['k-NN espacial', 'SELECT id, nombre, ubicacion FROM locales WHERE ubicacion KNN ((-77.0282, -12.1354), 10)'],
  ['k-NN vectorial (IVF)', "SET ivf.probes = 5"],
  ['imágenes por similitud', "SELECT id, categoria, archivo, emb <=> '[0.01, 0.02, 0.03]' AS dist\nFROM imagenes ORDER BY dist ASC LIMIT 5"],
  ['híbrida', "SELECT id, titulo, emb <=> '[0.01, 0.02, 0.03]' AS dist\nFROM documentos WHERE lang = 'es' ORDER BY dist ASC LIMIT 5"],
]

const VISTAS = [
  ['sql', 'Cliente SQL'],
  ['rag', 'Aplicación IA · RAG'],
  ['multimedia', 'Multimedia'],
]

const SQL_INICIAL = `CREATE TABLE empleados (
  id INT PRIMARY KEY,
  nombre CHAR(30),
  dept CHAR(20),
  salario FLOAT
) USING SEQUENTIAL`

export default function App() {
  const [sql, setSql] = useState(SQL_INICIAL)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [tables, setTables] = useState([])
  const [health, setHealth] = useState(null)
  const [selected, setSelected] = useState(null)
  const [pages, setPages] = useState(null)
  const [loadingPages, setLoadingPages] = useState(false)
  const [busy, setBusy] = useState(false)
  const [tab, setTab] = useState('resultados')
  const [vista, setVista] = useState('sql')

  const refrescar = useCallback(async () => {
    const [t, h] = await Promise.all([api.tables(), api.health()])
    setTables(t.tables)
    setHealth(h)
  }, [])

  useEffect(() => {
    refrescar().catch((e) => setError(e))
  }, [refrescar])

  const cargarPaginas = useCallback(async (tabla) => {
    if (!tabla) return setPages(null)
    setLoadingPages(true)
    try {
      setPages(await api.pages(tabla))
    } catch {
      setPages(null)
    } finally {
      setLoadingPages(false)
    }
  }, [])

  useEffect(() => {
    cargarPaginas(selected)
  }, [selected, cargarPaginas])

  async function ejecutar() {
    if (!sql.trim() || busy) return
    setBusy(true)
    setError(null)
    try {
      const r = await api.query(sql)
      setResult(r)
      setTab('resultados')
      await refrescar()
      if (r.plan.table) {
        setSelected(r.plan.table)
        await cargarPaginas(r.plan.table)
      }
    } catch (e) {
      setError(e)
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  async function reorganizar(tabla) {
    setBusy(true)
    setError(null)
    try {
      const r = await api.reorganize(tabla)
      setResult({
        columns: [],
        rows: [],
        row_count: 0,
        message:
          `Reorganización de «${tabla}»: ${r.stats.records.toLocaleString('es')} registros ` +
          `reescritos, ${r.before.pages} → ${r.after.pages} páginas, ` +
          `${r.before.overflow_records.toLocaleString('es')} registros sacados de overflow.`,
        plan: { access: 'DDL', table: tabla, reason: 'Fusión del área de overflow y reescritura del archivo principal al factor de carga.' },
        metrics: {
          disk_reads: r.stats.disk_reads,
          disk_writes: r.stats.disk_writes,
          parse_ms: 0,
          exec_ms: r.stats.elapsed_ms,
          total_ms: r.stats.elapsed_ms,
        },
      })
      await refrescar()
      await cargarPaginas(tabla)
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  async function sembrar(tabla) {
    const n = Number(window.prompt('¿Cuántos registros generar?', '10000'))
    if (!n || n < 1) return
    const info = tables.find((t) => t.name === tabla)
    const modo = info?.n_records ? 'insert' : 'bulk'
    setBusy(true)
    setError(null)
    try {
      const r = await api.seed(tabla, { n, mode: modo })
      setResult({
        columns: [],
        rows: [],
        row_count: 0,
        message:
          `${r.inserted.toLocaleString('es')} registros cargados en «${tabla}» por la ruta ` +
          `${modo === 'bulk' ? 'de carga masiva' : 'de inserción normal'}. ` +
          `La tabla tiene ahora ${r.state.n_records.toLocaleString('es')} registros en ` +
          `${r.state.pages} páginas, ${r.state.n_overflow.toLocaleString('es')} en overflow.`,
        plan: {
          access: 'DDL',
          table: tabla,
          reason:
            modo === 'bulk'
              ? 'Carga masiva: ordena y escribe páginas secuencialmente al factor de carga, sin leer.'
              : 'Inserción registro a registro: cada una localiza su página por búsqueda binaria.',
        },
        metrics: {
          disk_reads: r.metrics.disk_reads,
          disk_writes: r.metrics.disk_writes,
          parse_ms: 0,
          exec_ms: r.metrics.elapsed_ms,
          total_ms: r.metrics.elapsed_ms,
        },
      })
      await refrescar()
      await cargarPaginas(tabla)
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  function atajo(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault()
      ejecutar()
    }
  }

  return (
    <div className="shell">
      <header className="topbar">
        <h1>Mini-gestor de bases de datos multimodal</h1>
        <span className="course">CS2042 · Proyecto 2</span>
        <nav className="views" aria-label="Vistas">
          {VISTAS.map(([v, t]) => (
            <button key={v} aria-selected={vista === v} onClick={() => setVista(v)}>
              {t}
            </button>
          ))}
        </nav>
        {health && (
          <div className="facts">
            <span>bloque <b>{health.page_size / 1024} KB</b></span>
            <span>caché <b>{health.cache_size || 'sin'}</b></span>
            <span>tablas <b>{health.tables}</b></span>
          </div>
        )}
      </header>

      {vista === 'rag' && (
        <div className="workspace single">
          <RagPanel />
        </div>
      )}
      {vista === 'multimedia' && (
        <div className="workspace single">
          <MultimediaPanel />
        </div>
      )}
      <div className="workspace" hidden={vista !== 'sql'}>
        <aside className="panel sidebar">
          <header>
            Tablas
            <span className="count">{tables.length}</span>
          </header>
          <div className="scroll">
            <TableExplorer
              tables={tables}
              selected={selected}
              onSelect={setSelected}
              onReorganize={reorganizar}
              onSeed={sembrar}
              busy={busy}
            />
          </div>
        </aside>

        <div className="panel main">
          <section className="editor">
            <textarea
              value={sql}
              onChange={(e) => setSql(e.target.value)}
              onKeyDown={atajo}
              spellCheck={false}
              aria-label="Editor SQL"
              placeholder="SELECT * FROM empleados WHERE id = 101"
            />
            <div className="samples">
              {EJEMPLOS.map(([nombre, texto]) => (
                <button key={nombre} onClick={() => setSql(texto)}>
                  {nombre}
                </button>
              ))}
            </div>
            <div className="editor-bar">
              <button className="action primary" onClick={ejecutar} disabled={busy}>
                {busy ? 'Ejecutando' : 'Ejecutar'}
              </button>
              <span className="hint">
                <kbd>Ctrl</kbd> + <kbd>Enter</kbd> también ejecuta
              </span>
            </div>
          </section>

          <PlanMetrics result={result} />

          <section className="panel results">
            <header>
              <div className="tabs">
                <button
                  aria-selected={tab === 'resultados'}
                  onClick={() => setTab('resultados')}
                >
                  Resultados
                </button>
                <button aria-selected={tab === 'paginas'} onClick={() => setTab('paginas')}>
                  Mapa de páginas
                </button>
                {result?.spatial && (
                  <button aria-selected={tab === 'mapa'} onClick={() => setTab('mapa')}>
                    Mapa espacial
                  </button>
                )}
              </div>
              <span className="spacer" />
              {tab === 'resultados' && result?.columns?.length > 0 && (
                <span className="count">
                  {result.row_count.toLocaleString('es')} fila
                  {result.row_count === 1 ? '' : 's'}
                  {result.truncated && ` · se muestran ${result.rows.length}`}
                </span>
              )}
              {tab === 'paginas' && pages && (
                <span className="count">
                  {selected} · {pages.total_pages} página{pages.total_pages === 1 ? '' : 's'}
                </span>
              )}
              {tab === 'mapa' && result?.spatial && (
                <span className="count">
                  {result.spatial.query?.kind === 'knn'
                    ? `k=${result.spatial.query.k}`
                    : `radio ${result.spatial.query?.radius_km} km`}
                  {result.spatial.index_name
                    ? ` · ${result.spatial.index_name}`
                    : ' · sin índice'}
                </span>
              )}
            </header>
            <div className="scroll">
              {tab === 'resultados' ? (
                <ResultsTable result={result} error={error} />
              ) : tab === 'mapa' ? (
                <MapView result={result} />
              ) : (
                <PageMap data={pages} loading={loadingPages} />
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
