const ETIQUETA = {
  BinarySearch: 'Búsqueda binaria',
  BinaryRangeScan: 'Recorrido por rango',
  SeqScan: 'Escaneo secuencial',
  IndexScan: 'Búsqueda por índice',
  IndexRangeScan: 'Rango por índice',
  Insert: 'Inserción',
  Delete: 'Borrado',
  DDL: 'Definición',
  SET: 'Opción de sesión',
  KnnSeqScan: 'k-NN exacto',
  IVFScan: 'Búsqueda ANN (IVF)',
  PreFilter: 'Híbrido: pre-filtrado',
  PostFilter: 'Híbrido: post-filtrado',
  InvertedIndexScan: 'Índice invertido (MATCH)',
  TextSeqScan: 'Full Text Scan',
  TextPreFilter: 'Híbrido: texto → k-NN',
}

// Detalles del plan vectorial/hibrido que vale la pena mostrar, en orden.
const DETALLES = [
  ['probes', 'celdas sondeadas (probes)'],
  ['n_lists', 'celdas del IVF'],
  ['lists_scanned', 'listas leídas'],
  ['candidates_checked', 'candidatos revisados'],
  ['expansions', 'ampliaciones del sondeo'],
  ['stale_rids', 'RIDs obsoletos recuperados por PK'],
  ['prefilter_access', 'acceso del pre-filtro'],
  ['residual_filters', 'filtros residuales'],
  ['sort', 'ordenamiento'],
  ['forced', 'forzado'],
]

// Detalles del MATCH (plan.text), en orden.
const DETALLES_TEXTO = [
  ['mode', 'tipo de consulta', (v) => (v === 'boolean' ? 'booleana' : 'ranking (OR)')],
  ['terms', 'términos (raíces)'],
  ['matches', 'documentos que cumplen'],
  ['lists_read', 'listas de posteo leídas'],
  ['postings_read', 'postings leídos'],
  ['rows_fetched', 'filas traídas de la tabla'],
  ['scoring', 'ranking'],
  ['rebuilt', 'mantenimiento'],
]

function IoSplit({ io }) {
  const tope = Math.max(io.index_reads, io.table_reads, 1)
  return (
    <div className="compare">
      {[
        ['índice', io.index_reads, 'fill'],
        ['tabla', io.table_reads, 'fill alt'],
      ].map(([nombre, n, clase]) => (
        <div className="bar-row" key={nombre}>
          <span>lecturas {nombre}</span>
          <span className="track">
            <span className={clase} style={{ width: `${(n / tope) * 100}%` }} />
          </span>
          <span className="num">{n.toLocaleString('es')}</span>
        </div>
      ))}
    </div>
  )
}

function claseAcceso(access) {
  if (access === 'SeqScan') return 'access scan'
  if (access === 'TextSeqScan') return 'access scan'
  if (access.startsWith('Index') || access === 'IVFScan' || access === 'InvertedIndexScan') return 'access pending'
  return 'access'
}

export default function PlanMetrics({ result }) {
  if (!result) {
    return (
      <section className="telemetry">
        <div className="plan">
          <span className="access scan">Sin ejecutar</span>
          <p>
            El plan de ejecución y el desglose de bloques aparecerán aquí después de la
            primera consulta.
          </p>
        </div>
        <div className="readout">
          {['bloques leídos', 'bloques escritos', 'parseo', 'ejecución'].map((u) => (
            <div key={u}>
              <div className="value">—</div>
              <div className="unit">{u}</div>
            </div>
          ))}
        </div>
      </section>
    )
  }

  const { plan, metrics } = result
  const leidos = metrics.disk_reads
  const fullScan = plan.pages ?? null
  // Solo tiene sentido comparar cuando la ruta elegida NO es el escaneo completo:
  // es ahi donde el planificador esta afirmando haber ahorrado I/O.
  const comparable = fullScan && !['SeqScan', 'KnnSeqScan', 'TextSeqScan'].includes(plan.access) && leidos > 0
  const valor = (v) => (Array.isArray(v) ? v.join(', ') : String(v))
  const detalles = DETALLES.filter(([k]) => plan[k] !== undefined && plan[k] !== null).map(
    ([k, etiqueta]) => [k, etiqueta, valor(plan[k])],
  )
  if (plan.text) {
    for (const [k, etiqueta, fmt] of DETALLES_TEXTO) {
      const v = plan.text[k]
      if (v !== undefined && v !== null) detalles.push([`text.${k}`, etiqueta, fmt ? fmt(v) : valor(v)])
    }
  }
  const tope = comparable ? Math.max(leidos, fullScan) : 0

  return (
    <section className="telemetry">
      <div className="plan">
        <span className={claseAcceso(plan.access)}>
          {ETIQUETA[plan.access] ?? plan.access}
        </span>
        <p>{plan.reason}</p>
        {plan.decision && <p className="ratio">Decisión del planificador: {plan.decision}</p>}
        {plan.io && (plan.io.index_reads > 0 || plan.io.table_reads > 0) && <IoSplit io={plan.io} />}
        {detalles.length > 0 && (
          <dl className="plan-facts">
            {detalles.map(([k, etiqueta, v]) => (
              <div key={k}>
                <dt>{etiqueta}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
        )}

        {comparable && (
          <div className="compare">
            <div className="bar-row">
              <span>ruta elegida</span>
              <span className="track">
                <span className="fill" style={{ width: `${(leidos / tope) * 100}%` }} />
              </span>
              <span className="num">{leidos}</span>
            </div>
            <div className="bar-row">
              <span>escaneo completo</span>
              <span className="track">
                <span className="fill alt" style={{ width: '100%' }} />
              </span>
              <span className="num">{fullScan}</span>
            </div>
            <p className="ratio">
              <b>{(fullScan / leidos).toFixed(0)}×</b> menos bloques que recorrer la tabla
              entera
            </p>
          </div>
        )}
      </div>

      <div className="readout">
        <div>
          <div className="value">{metrics.disk_reads.toLocaleString('es')}</div>
          <div className="unit">bloques leídos</div>
        </div>
        <div>
          <div className={metrics.disk_writes ? 'value writes' : 'value'}>
            {metrics.disk_writes.toLocaleString('es')}
          </div>
          <div className="unit">bloques escritos</div>
        </div>
        <div>
          <div className="value">{metrics.parse_ms.toFixed(2)}</div>
          <div className="unit">ms de parseo</div>
        </div>
        <div>
          <div className="value">{metrics.exec_ms.toFixed(2)}</div>
          <div className="unit">ms de ejecución</div>
        </div>
      </div>
    </section>
  )
}
