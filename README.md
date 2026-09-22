# Proyecto 1 — Gestor de Bases de Datos Multimodal (CS2042)

Capa de memoria secundaria y organización **Sequential File**.

## Levantar el sistema

```bash
pip install fastapi uvicorn pydantic pytest
cd frontend && npm install && npm run build && cd ..
python -m uvicorn backend.api.main:app --port 8000
```

Abre <http://127.0.0.1:8000>. FastAPI sirve el bundle de React, así que es un solo
proceso y una sola URL.

Para desarrollar el frontend con recarga en caliente, en otra terminal:
`cd frontend && npm run dev` (el dev server proxea `/api` al backend del 8000).

Tests: `python -m pytest -q`

Variables de entorno: `BD2_DATA_DIR` (por defecto `data`), `BD2_PAGE_SIZE` (4096) y
`BD2_CACHE_SIZE` (0 = sin buffer pool, que es como deben correr los benchmarks).

## SQL soportado

```sql
CREATE TABLE empleados (id INT PRIMARY KEY, nombre CHAR(30), salario FLOAT) USING SEQUENTIAL;
INSERT INTO empleados VALUES (101, 'Ada Lovelace', 5200.0);
SELECT * FROM empleados WHERE id = 101;                 -- búsqueda binaria
SELECT * FROM empleados WHERE id >= 100 AND id <= 500;  -- recorrido por rango
SELECT * FROM empleados WHERE nombre = 'Ada';           -- escaneo secuencial
DELETE FROM empleados WHERE id = 101;
DROP TABLE empleados;
```

`USING HEAP` y `USING SEQUENTIAL` funcionan. `CREATE INDEX ... USING HASH` construye
un Hashing Dinámico sobre disco y habilita `IndexScan` en el planificador.
`CREATE INDEX ... USING BTREE` se parsea, pero devuelve 501: el árbol B+ sigue
pendiente.

## API REST

| Método | Ruta | Estado |
|---|---|---|
| `POST` | `/api/query` | funcionando |
| `GET` | `/api/tables` | funcionando |
| `GET` | `/api/tables/{t}` | funcionando |
| `POST` | `/api/tables/{t}/reorganize` | funcionando |
| `GET` | `/api/tables/{t}/pages` | funcionando — inspección de páginas físicas |
| `POST` | `/api/tables/{t}/seed` | funcionando — carga de datos sintéticos |
| `GET` | `/api/health` | funcionando |
| `POST` | `/api/benchmarks/run` | **501** — necesita B+ (usa Heap/Sequential/Hash mientras tanto) |
| `GET` | `/api/tables/{t}/index/{n}` | **501** — inspección de índice, aún no implementada |

Documentación interactiva en `/docs`.

## Módulos

### `backend/storage/` — capa de almacenamiento físico

| Módulo | Responsabilidad |
|---|---|
| `schema.py` | Tipos `INT`/`FLOAT`/`CHAR(n)`, `pack`/`unpack` con `struct`, extracción rápida de la PK |
| `disk_counter.py` | `disk_reads`/`disk_writes` + `measure()` para métricas por consulta |
| `page.py` | Cabecera de 32 B, bitmap de presencia, búsqueda binaria intra-página, `low_key` |
| `pager.py` | Página 0 autodescriptiva, free-list, I/O por offset (`pread`/`lseek`) |
| `buffer_pool.py` | Caché LRU write-back, desactivada por defecto |

### `backend/structures/` — organización de archivos

| Módulo | Responsabilidad | Estado |
|---|---|---|
| `sequential_file.py` | Área principal ordenada, overflow encadenado, reorganización | listo |
| `heap_file.py` | Free-list y move-the-last, full scan | listo (luism) |

### `backend/index/` — índices

| Módulo | Responsabilidad | Estado |
|---|---|---|
| `btree.py` | Árbol B+ multinivel en disco | **pendiente** |
| `hash.py` | Extendible Hashing | listo (sebastian) |

### `backend/engine/` — motor de consultas

| Módulo | Responsabilidad |
|---|---|
| `parser.py` | Parser SQL por descenso recursivo, sin librerías |
| `catalog.py` | Catálogo reconstruido leyendo la página 0 de cada archivo |
| `executor.py` | Planificador de ruta de acceso y ejecución con telemetría |
| `errors.py` | Jerarquía de errores que la API mapea a códigos HTTP |

### `frontend/` — cliente SQL

React + Vite. Cuatro paneles: explorador de tablas, editor SQL, visor de resultados y
plan de ejecución con métricas de I/O. Incluye un mapa de páginas físicas que dibuja la
ocupación real de cada bloque y sus cadenas de overflow.

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

### Hash Extensible

Dos archivos por índice (`<tabla>__<indice>.hash.dir` y `.hash.bkt`), mismo
patrón que el par `.seq.dat`/`.seq.ovf`. `pager.user_a` del archivo `.dir` es
la profundidad global; `page.aux_page_id` de cada bucket es su profundidad
local. Un bucket que no logra separarse al dividirse (claves muy repetidas)
encadena overflow por `next_page_id` en vez de seguir duplicando el
directorio. `Catalog.create_index`/`open_index` lo conectan al motor;
`INSERT`/`DELETE` mantienen el índice al día.

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

## Qué falta para cerrar el Proyecto 1

- `BPlusTree` en `backend/index/btree.py`
- Suite de los 4 experimentos (`POST /api/benchmarks/run`)

Cada archivo pendiente documenta sus requisitos del enunciado y qué le da ya la capa de
almacenamiento. Para conectar un índice nuevo al motor basta con registrarlo en
`Catalog.indexes_of()` e implementar su rama en `Executor._rows_for()`: el planificador
ya elige `IndexScan` e `IndexRangeScan` en cuanto detecta el índice.

## Limitaciones conocidas

- `SequentialFile` exige clave primaria `INT` o `FLOAT`: el `low_key` de la cabecera de
  página ocupa 8 bytes. Una PK `CHAR(n)` levanta `UnsupportedKeyType`.
- `reorganize()` invalida los `RID` previamente entregados, igual que cualquier
  reorganización de archivo.
