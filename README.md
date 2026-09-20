# Proyecto 1 — Gestor de Bases de Datos Multimodal (CS2042)

Capa de memoria secundaria y organización **Sequential File**.

## Ejecutar

```bash
pip install pytest
python -m pytest -q
```

## Módulos

### `backend/storage/` — capa de almacenamiento físico

| Módulo | Responsabilidad |
|---|---|
| `schema.py` | Tipos `INT`/`FLOAT`/`CHAR(n)`, `pack`/`unpack` con `struct`, extracción rápida de la PK |
| `disk_counter.py` | `disk_reads`/`disk_writes` + `measure()` para métricas por consulta |
| `page.py` | Cabecera de 32 B, bitmap de presencia, búsqueda binaria intra-página, `low_key` |
| `pager.py` | Página 0 autodescriptiva, free-list, I/O por offset (`pread`/`lseek`) |
| `buffer_pool.py` | Caché LRU write-back, desactivada por defecto |

### `backend/organization/` — organización de archivos

| Módulo | Responsabilidad |
|---|---|
| `sequential_file.py` | Área principal ordenada, overflow encadenado por página, reorganización |

## Contrato con el resto del equipo

**Ningún módulo fuera de `backend/storage/` abre archivos de datos o índices.** Todo
acceso a disco pasa por `Pager`. Si esa regla se rompe, el `DiskCounter` deja de ser
confiable y los cuatro experimentos del informe quedan invalidados.

```python
from backend.storage.pager import Pager
from backend.storage.page import PageType

pg = Pager("tabla.idx", record_size=12, page_size=4096, counter=counter_compartido)
page = pg.new_page(PageType.BT_LEAF)   # asigna sin leer de vuelta
pg.write_page(page)                     # counter.disk_writes += 1
page = pg.read_page(page.page_id)       # counter.disk_reads  += 1
```

### Para quien implemente el B+ Tree

- `page.next_page_id` / `page.prev_page_id` de la cabecera **ya son** los
  `next_leaf_id` / `prev_leaf_id` que pide el enunciado para el recorrido por rango.
- `pager.user_a` / `pager.user_b` son dos `uint32` persistentes en la cabecera del
  archivo: usa `user_a` como `root_page_id`.
- `page_capacity(page_size, record_size)` da el fan-out directamente, para la tabla del
  Experimento 4.

### Para quien implemente el Hash Extensible

- `pager.user_a` sirve como profundidad global; `page.aux_page_id` como profundidad
  local del bucket.
- `pager.free_page(id)` devuelve buckets vacíos a la free-list tras una fusión.

## Decisiones de diseño

| Decisión | Elección | Fuente |
|---|---|---|
| Organización de página | Longitud fija + bitmap | El SQL del Entregable 1 solo usa tipos de ancho fijo |
| Overflow | Encadenado **por página**, una cadena por página principal | En un sistema paginado, encadenar registros cuesta 1 I/O por registro |
| Cadena de overflow | Ordenada por clave | Clase 04, p. 10 |
| Umbral de reorganización | `K = (C/2)·⌈log₂(P)⌉` registros | Clase 04, p. 9 («límite máximo de K registros») + tabla de costos p. 59 |
| Fill factor | 0.75 uniforme | Enunciado 3.2.2 (70–80 %) |
| Buffer pool | Opcional, apagado por defecto | Benchmarks sin caché validan la teoría; la demo con caché va rápida |

Detalle completo en `docs/superpowers/specs/` y `docs/superpowers/plans/`.

## Limitaciones conocidas

- `SequentialFile` exige clave primaria `INT` o `FLOAT`: el `low_key` de la cabecera de
  página ocupa 8 bytes. Una PK `CHAR(n)` levanta `UnsupportedKeyType`.
- `reorganize()` invalida los `RID` previamente entregados, igual que cualquier
  reorganización de archivo.
