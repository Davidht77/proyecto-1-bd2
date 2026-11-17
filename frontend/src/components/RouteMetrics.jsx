const ETIQUETA = {
  InvertedIndex: 'Índice invertido',
  IVFScan: 'Búsqueda ANN (IVF)',
  KnnSeqScan: 'k-NN exacto',
  PreFilter: 'Híbrido: pre-filtrado',
  PostFilter: 'Híbrido: post-filtrado',
}

/**
 * Desglose de una ruta de recuperacion: ruta de acceso, latencia y bloques
 * leidos en el indice y en la tabla. Es la misma informacion que el panel de
 * metricas del cliente SQL, compactada para poder poner dos rutas lado a lado.
 */
export default function RouteMetrics({ metrics, extra = [] }) {
  if (!metrics) return null
  const idx = metrics.index_reads ?? 0
  const tab = metrics.table_reads ?? 0
  const tope = Math.max(idx, tab, 1)
  return (
    <div className="route-metrics">
      <div className="route-head">
        <span className="access">{ETIQUETA[metrics.access] ?? metrics.access}</span>
        <span className="num">{metrics.exec_ms?.toFixed(2)} ms</span>
      </div>
      <div className="compare">
        {[
          ['lecturas índice', idx, 'fill'],
          ['lecturas tabla', tab, 'fill alt'],
        ].map(([nombre, n, clase]) => (
          <div className="bar-row" key={nombre}>
            <span>{nombre}</span>
            <span className="track">
              <span className={clase} style={{ width: `${(n / tope) * 100}%` }} />
            </span>
            <span className="num">{n.toLocaleString('es')}</span>
          </div>
        ))}
      </div>
      <dl className="plan-facts">
        <div>
          <dt>bloques leídos</dt>
          <dd>{(metrics.disk_reads ?? idx + tab).toLocaleString('es')}</dd>
        </div>
        {metrics.probes != null && (
          <div>
            <dt>celdas sondeadas</dt>
            <dd>
              {metrics.probes} / {metrics.n_lists}
            </dd>
          </div>
        )}
        {metrics.terms != null && (
          <div>
            <dt>términos de la consulta</dt>
            <dd>{metrics.terms}</dd>
          </div>
        )}
        {metrics.embed_ms != null && (
          <div>
            <dt>embedding de la consulta</dt>
            <dd>{metrics.embed_ms.toFixed(1)} ms</dd>
          </div>
        )}
        {extra.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
