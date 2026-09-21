export default function ResultsTable({ result, error }) {
  if (error) {
    const pendiente = error.kind === 'no_implementado'
    return (
      <div className={pendiente ? 'notice pending' : 'notice error'}>
        <h3>{pendiente ? 'Todavía no construido' : 'La consulta no se pudo ejecutar'}</h3>
        <p>{error.message}</p>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="empty">
        <h3>Escribe una consulta</h3>
        <p>
          Los resultados se muestran aquí, y arriba verás por qué ruta de acceso pasó la
          consulta y cuántos bloques físicos costó.
        </p>
      </div>
    )
  }

  if (!result.columns.length) {
    return (
      <div className="notice ok">
        <h3>Listo</h3>
        <p>{result.message}</p>
      </div>
    )
  }

  if (!result.rows.length) {
    return (
      <div className="empty">
        <h3>Ninguna fila coincide</h3>
        <p>La consulta se ejecutó correctamente, pero el predicado no seleccionó registros.</p>
      </div>
    )
  }

  return (
    <table className="grid">
      <thead>
        <tr>
          {result.columns.map((c) => (
            <th key={c}>{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {result.rows.map((fila, i) => (
          <tr key={i}>
            {fila.map((v, j) => (
              <td key={j} className={typeof v === 'number' ? 'num' : ''}>
                {typeof v === 'number' ? v.toLocaleString('es') : v}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
