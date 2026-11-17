import { useEffect, useRef } from 'react'
import L from 'leaflet'

/**
 * Visor espacial: dibuja los puntos devueltos por una consulta WITHIN_RADIUS o
 * KNN, el circulo de busqueda y los MBRs que el R-Tree tuvo que inspeccionar.
 *
 * Se usa Leaflet directamente (sin react-leaflet) para no agregar otra capa de
 * dependencias. El mapa se inicializa una sola vez; cada resultado reemplaza
 * solo la capa de overlays.
 */
export default function MapView({ result }) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const layerRef = useRef(null)

  useEffect(() => {
    if (mapRef.current || !containerRef.current) return
    const map = L.map(containerRef.current, {
      center: [-12.1354, -77.0282],
      zoom: 12,
      zoomControl: true,
    })
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap',
    }).addTo(map)
    mapRef.current = map
    layerRef.current = L.layerGroup().addTo(map)
    return () => {
      map.remove()
      mapRef.current = null
      layerRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    const layer = layerRef.current
    if (!map || !layer || !result) return

    layer.clearLayers()
    const columns = result.columns ?? []
    const pointColumn = result.spatial?.point_column
    const pIndex = columns.findIndex((c) => c === pointColumn)
    const query = result.spatial?.query
    const bounds = []

    if (pIndex >= 0) {
      for (const row of result.rows ?? []) {
        const point = row[pIndex]
        if (!Array.isArray(point) || point.length < 2) continue
        const [lon, lat] = point
        L.circleMarker([lat, lon], {
          radius: 5,
          color: '#1d6f8f',
          weight: 1,
          fillColor: '#2d9cdb',
          fillOpacity: 0.75,
        })
          .bindPopup(`(${lon.toFixed(5)}, ${lat.toFixed(5)})`)
          .addTo(layer)
        bounds.push([lat, lon])
      }
    }

    if (query?.center?.length >= 2) {
      const [lon, lat] = query.center
      bounds.push([lat, lon])
      if (query.kind === 'radius') {
        L.circle([lat, lon], {
          radius: (query.radius_km ?? 0) * 1000,
          color: '#d9822b',
          weight: 2,
          fillColor: '#f2b46b',
          fillOpacity: 0.12,
        }).addTo(layer)
      } else {
        L.circleMarker([lat, lon], {
          radius: 7,
          color: '#d9822b',
          weight: 2,
          fillColor: '#fff',
          fillOpacity: 1,
        })
          .bindPopup(`Centro KNN (k=${query.k ?? ''})`)
          .addTo(layer)
      }
    }

    for (const mbr of result.spatial?.visited_mbrs ?? []) {
      if (!Array.isArray(mbr) || mbr.length < 4) continue
      const [xmin, ymin, xmax, ymax] = mbr
      L.rectangle(
        [
          [ymin, xmin],
          [ymax, xmax],
        ],
        {
          color: '#8a6d3b',
          weight: 1,
          fill: false,
          dashArray: '4 3',
          opacity: 0.7,
        },
      ).addTo(layer)
    }

    if (bounds.length === 1) {
      map.setView(bounds[0], 13)
    } else if (bounds.length > 1) {
      map.fitBounds(bounds, { padding: [24, 24], maxZoom: 14 })
    }
  }, [result])

  const pointColumn = result?.spatial?.point_column
  const hasPointColumn =
    pointColumn && (result?.columns ?? []).some((c) => c === pointColumn)

  return (
    <div className="mapview">
      <div ref={containerRef} className="mapview-canvas" />
      <div className="mapview-legend">
        {!hasPointColumn && (
          <p className="notice pending">
            La consulta no proyecta la columna espacial «{pointColumn ?? '—'}». Inclúyela en
            el SELECT para ver los puntos.
          </p>
        )}
        <span><i className="dot point" /> resultados</span>
        {result?.spatial?.query?.kind === 'radius' && (
          <span><i className="dot circle" /> radio de busqueda</span>
        )}
        <span><i className="dot mbr" /> MBRs inspeccionados</span>
      </div>
    </div>
  )
}
