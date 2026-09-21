import { useState } from 'react'

const TIPO = (c) => (c.type === 'CHAR' ? `char(${c.length})` : c.type.toLowerCase())

function bytes(n) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(2)} MB`
}

export default function TableExplorer({ tables, selected, onSelect, onReorganize, onSeed, busy }) {
  const [abierta, setAbierta] = useState(selected)

  function alternar(nombre) {
    const siguiente = abierta === nombre ? null : nombre
    setAbierta(siguiente)
    if (siguiente) onSelect(siguiente)
  }

  if (!tables.length) {
    return (
      <div className="empty">
        <h3>Todavía no hay tablas</h3>
        <p>
          Crea una desde el editor con <code>CREATE TABLE</code> y aparecerá aquí con su
          esquema y su ocupación en disco.
        </p>
      </div>
    )
  }

  return (
    <div>
      {tables.map((t) => {
        const activa = abierta === t.name
        return (
          <div className="table-item" key={t.name}>
            <button
              className="table-head"
              aria-expanded={activa}
              onClick={() => alternar(t.name)}
            >
              <span className="name">{t.name}</span>
              <span className="rows">{t.n_records.toLocaleString('es')}</span>
            </button>

            {activa && (
              <div className="table-body">
                <ul className="cols">
                  {t.columns.map((c) => (
                    <li key={c.name}>
                      <span>{c.name}</span>
                      <span className="type">{TIPO(c)}</span>
                      {c.primary_key && <span className="pk">clave</span>}
                    </li>
                  ))}
                </ul>

                <div className="stat-grid">
                  <div><span>motor</span><b>{t.engine.toLowerCase()}</b></div>
                  <div><span>páginas</span><b>{t.page_count}</b></div>
                  <div><span>registro</span><b>{t.record_size} B</b></div>
                  <div><span>bloque</span><b>{t.page_size / 1024} KB</b></div>
                  {t.engine === 'SEQUENTIAL' && (
                    <div className={t.n_overflow ? 'warn' : ''}>
                      <span>overflow</span><b>{t.n_overflow}</b>
                    </div>
                  )}
                  <div><span>en disco</span><b>{bytes(t.size_bytes)}</b></div>
                </div>

                <div className="row-actions">
                  {t.engine === 'SEQUENTIAL' && (
                    <button
                      className="action"
                      disabled={busy || !t.n_overflow}
                      onClick={() => onReorganize(t.name)}
                      title={
                        t.n_overflow
                          ? 'Fusiona el área de overflow y reescribe el archivo principal'
                          : 'No hay nada en overflow que fusionar'
                      }
                    >
                      Reorganizar
                    </button>
                  )}
                  <button className="action" disabled={busy} onClick={() => onSeed(t.name)}>
                    Cargar datos
                  </button>
                </div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
