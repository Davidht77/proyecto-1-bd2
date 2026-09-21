import { useCallback, useEffect, useState } from 'react'
import { api } from './lib/api'
import TableExplorer from './components/TableExplorer'
import PlanMetrics from './components/PlanMetrics'
import ResultsTable from './components/ResultsTable'
import PageMap from './components/PageMap'

const EJEMPLOS = [
  ['crear tabla', "CREATE TABLE empleados (\n  id INT PRIMARY KEY,\n  nombre CHAR(30),\n  dept CHAR(20),\n  salario FLOAT\n) USING SEQUENTIAL"],
  ['insertar', "INSERT INTO empleados VALUES (101, 'Ada Lovelace', 'Analytics', 5200.0)"],
  ['igualdad', 'SELECT * FROM empleados WHERE id = 101'],
  ['rango', 'SELECT * FROM empleados WHERE id >= 100 AND id <= 500'],
  ['escaneo completo', "SELECT * FROM empleados WHERE dept = 'Analytics'"],
  ['borrar', 'DELETE FROM empleados WHERE id = 101'],
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
        <h1>Mini-gestor de bases de datos</h1>
        <span className="course">CS2042 · Proyecto 1</span>
        {health && (
          <div className="facts">
            <span>bloque <b>{health.page_size / 1024} KB</b></span>
            <span>caché <b>{health.cache_size || 'sin'}</b></span>
            <span>tablas <b>{health.tables}</b></span>
          </div>
        )}
      </header>

      <div className="workspace">
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
            </header>
            <div className="scroll">
              {tab === 'resultados' ? (
                <ResultsTable result={result} error={error} />
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
