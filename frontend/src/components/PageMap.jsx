/**
 * Mapa de páginas físicas.
 *
 * Una celda por página del área principal: la altura del relleno es su
 * ocupación real, y las marcas que cuelgan debajo son las páginas de su cadena
 * de overflow. Es la estructura de datos en disco, dibujada tal cual.
 */
export default function PageMap({ data, loading }) {
  if (loading) {
    return <div className="empty"><p>Leyendo cabeceras de página…</p></div>
  }

  if (!data) {
    return (
      <div className="empty">
        <h3>Selecciona una tabla</h3>
        <p>
          Aquí se dibuja cada página física del archivo: cuánto la ocupan sus registros y
          qué cadena de overflow le cuelga.
        </p>
      </div>
    )
  }

  if (!data.pages.length) {
    return (
      <div className="empty">
        <h3>La tabla está vacía</h3>
        <p>Aún no se ha asignado ninguna página. Inserta registros o usa «Cargar datos».</p>
      </div>
    )
  }

  // Las cifras se reportan sobre la ventana cargada, no sobre el total, porque
  // solo se leyeron las cabeceras de estas páginas. El overflow global sí viene
  // del contador del archivo.
  const visibles = data.pages.length
  const parcial = visibles < data.total_pages
  const conCadena = data.pages.filter((p) => p.overflow_chain.length).length
  const ocupacion = data.pages.reduce((a, p) => a + p.fill_pct, 0) / visibles

  return (
    <div className="pagemap">
      <div className="strip">
        {data.pages.map((p) => (
          <div
            className="cell"
            key={p.page_id}
            title={
              `página ${p.logical_index} (bloque físico ${p.page_id})\n` +
              `${p.record_count} de ${p.capacity} registros · ${p.fill_pct}%\n` +
              `claves ${p.first_key ?? '—'} … ${p.last_key ?? '—'}\n` +
              `rango desde ${p.low_key === null ? '−∞' : p.low_key}` +
              (p.overflow_chain.length
                ? `\noverflow: ${p.overflow_chain.length} página(s), ` +
                  `${p.overflow_chain.reduce((a, o) => a + o.record_count, 0)} registros`
                : '')
            }
          >
            <span className="main" style={{ height: 56 }}>
              <i style={{ height: `${p.fill_pct}%` }} />
            </span>
            {p.overflow_chain.length > 0 && (
              <span className="chain">
                {p.overflow_chain.map((o) => (
                  <span key={o.page_id} />
                ))}
              </span>
            )}
          </div>
        ))}
      </div>

      {parcial && (
        <p className="window-note">
          Se dibujan las primeras {visibles} páginas de {data.total_pages.toLocaleString('es')}.
        </p>
      )}

      <div className="legend">
        <span>
          <i className="s" />
          {ocupacion.toFixed(0)}% de ocupación media
          {parcial ? ' en las páginas dibujadas' : ''}
        </span>
        <span>
          <i className="o" />
          {data.overflow_records.toLocaleString('es')} registros en overflow
          {conCadena > 0 && `, ${conCadena} de estas páginas con cadena`}
        </span>
        <span>
          bloque de {data.page_size / 1024} KB · {data.capacity_per_page} registros por página ·
          reorganiza al superar {data.overflow_cap.toLocaleString('es')}
        </span>
      </div>
    </div>
  )
}
