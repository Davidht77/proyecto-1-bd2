# Gestor de Bases de Datos Multimodal (CS2042) — Proyectos 1 y 2

Motor relacional sobre memoria secundaria (Heap, Sequential, B+, Hash) extendido con
los módulos multimodales del Entregable 2: espacial (R-Tree), textual (índice
invertido), vectorial (IVF) y la **aplicación de IA** (RAG sobre Wikipedia) con el
módulo multimedia (BoVW vs. embeddings densos).

## Levantar el sistema (un solo paso)

```bash
./run.sh          # Linux / macOS / Git Bash
.\run.ps1         # Windows PowerShell
```

El script crea `.venv`, instala `requirements.txt`, construye el frontend y levanta la
API. Requisitos: Python ≥ 3.11 y Node ≥ 18. Equivale a:

```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
python -m uvicorn backend.api.main:app --port 8000
```

Abre <http://127.0.0.1:8000>. FastAPI sirve el bundle de React, así que es un solo
proceso y una sola URL. La barra superior cambia entre el **Cliente SQL**, la
**Aplicación IA · RAG** y la vista **Multimedia**.

Para dejar la aplicación de IA lista (datasets, corpus e índices) ver la sección
[Aplicación de IA y módulo multimedia](#aplicación-de-ia-y-módulo-multimedia-rol-4).

Para desarrollar el frontend con recarga en caliente, en otra terminal:
`cd frontend && npm run dev` (el dev server proxea `/api` al backend del 8000).

Tests: `python -m pytest -q`

Variables de entorno: `BD2_DATA_DIR` (por defecto `data`), `BD2_PAGE_SIZE` (4096) y
`BD2_CACHE_SIZE` (0 = sin buffer pool, que es como deben correr los benchmarks).

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `backend/` | motor (almacenamiento, archivos, índices, SQL, módulos textual/vectorial/multimedia, RAG) y API REST (FastAPI) |
| `frontend/` | cliente web React + Vite (SQL, mapa Leaflet, RAG, multimedia) |
| `data/` | archivos binarios del motor y datasets descargados (ver `data/README.md`) |
| `benchmarks/results/` | CSV de los Experimentos 1–8; el código está en `backend/benchmarks/` |
| `informe_analisis/` | informes LaTeX: `informe_analisis.tex` (P1) e `informe_p2.tex` (P2), con sus PDF |
| `tests/` | `python -m pytest -q` |

## Cobertura del enunciado (Entregable 2)

| Enunciado | Dónde |
|---|---|
| 5.1 R-Tree en disco: MBR, split cuadrático, rango/radio, k-NN por `MINDIST`, mapa Leaflet | `backend/index/rtree.py`, `geo.py`, `frontend/src/components/MapView.jsx` |
| 5.2 Índice invertido: tokenización, stopwords, stemming Snowball, SPIMI, `vocab.dic` + `postings.inv`, booleano AND/OR/AND-NOT en O(n+m), TF-IDF + coseno (normas L2 en disco), BM25 | `backend/text/` |
| 5.3 BoVW (SIFT + K-Means + TF-IDF sobre el índice invertido), embeddings densos (ResNet-18, SBERT), k-NN exacto con Max-Heap, IVF en disco | `backend/multimedia/`, `backend/vector/` |
| 5.4 `USING RTREE \| INVERTED \| IVF`, `WITHIN_RADIUS`, `KNN`, `MATCH`, `COSINE_SCORE()`, `<->`, `<=>`, `SET ivf.probes`, planificador PRE/POST con I/O de índice vs tabla | `backend/engine/` |
| 5.5 Aplicación de IA: RAG (opción A) sobre 15 000 artículos, léxico vs semántico, LLM con fuentes y métricas de I/O | `backend/rag/`, `frontend/src/components/RagPanel.jsx` |
| 6 Experimentos 5–8 | `backend/benchmarks/experiment{5,6,7,8}.py` → `benchmarks/results/` → `informe_analisis/figuras/generar_p2.py` |

HNSW (opción 2 de 5.3) no se implementó: el enunciado pide al menos una estructura ANN y
se eligió IVF.

## SQL soportado

```sql
CREATE TABLE empleados (id INT PRIMARY KEY, nombre CHAR(30), salario FLOAT) USING SEQUENTIAL;
INSERT INTO empleados VALUES (101, 'Ada Lovelace', 5200.0);
SELECT * FROM empleados WHERE id = 101;                 -- búsqueda binaria
SELECT * FROM empleados WHERE id >= 100 AND id <= 500;  -- recorrido por rango
SELECT * FROM empleados WHERE nombre = 'Ada';           -- escaneo secuencial
DELETE FROM empleados WHERE id = 101;
DROP TABLE empleados;

-- módulo espacial
CREATE TABLE locales (id INT PRIMARY KEY, nombre CHAR(20), ubicacion POINT) USING HEAP;
INSERT INTO locales VALUES (1, 'Miraflores', POINT(-77.0282, -12.1354));
CREATE INDEX idx_geo ON locales (ubicacion) USING RTREE;
SELECT * FROM locales WHERE ubicacion WITHIN_RADIUS ((-77.0282, -12.1354), 2.5);
SELECT * FROM locales WHERE ubicacion KNN ((-77.0282, -12.1354), 10);

-- módulo textual
CREATE INDEX idx_texto ON documentos (texto) USING INVERTED WITH (scoring = 'TFIDF_COSINE');
SELECT id, titulo, COSINE_SCORE() AS relevancia FROM documentos
WHERE texto MATCH 'aprendizaje AND profundo AND NOT basico'
ORDER BY relevancia DESC LIMIT 10;
```

`USING HEAP` y `USING SEQUENTIAL` funcionan. `CREATE INDEX ... USING HASH` construye
un Hashing Dinámico sobre disco y habilita `IndexScan` en el planificador.
`CREATE INDEX ... USING BTREE` construye el árbol B+; `USING RTREE` indexa columnas
`POINT`; `USING INVERTED` indexa columnas de texto para `MATCH`. `WITHIN_RADIUS`,
`KNN` y `MATCH` se ejecutan con su índice o, si no hay, con un recorrido de la tabla.

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
| `POST` | `/api/benchmarks/run` | funcionando — Experimentos 1–5 (`{"experiment": 0}` = todos) |
| `GET` | `/api/tables/{t}/index/{n}` | funcionando para `RTREE` (altura, MBR y nodos); el resto se inspecciona desde el plan |
| `GET`/`POST` | `/api/rag/*` | aplicación RAG (estado, construcción, pregunta) |
| `GET`/`POST` | `/api/multimedia/*` | tienda de imágenes (BoVW vs ResNet-18 + IVF) |

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
| `btree.py` | Árbol B+ multinivel en disco (split, punto, rango por `next_leaf`) | listo |
| `hash.py` | Extendible Hashing | listo (sebastian) |
| `rtree.py` | R-Tree espacial persistido (MBR, split cuadrático, k-NN) | listo |
| `geo.py` | Haversine, caja de radio y `MINDIST` geográfico | listo |

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

## Proyecto 2 — Búsqueda vectorial ANN y motor SQL híbrido

Enunciado 5.3 (búsqueda exacta + IVF en disco) y 5.4 (gramática multimodal y
planificador híbrido). Código en `backend/vector/` y `backend/engine/`.

### SQL

```sql
CREATE TABLE multimedia (id INT PRIMARY KEY, categoria CHAR(20), precio FLOAT,
                         embedding VECTOR(128)) USING HEAP;
INSERT INTO multimedia VALUES (1, 'footwear', 120.0, '[0.12, -0.34, ...]');
CREATE INDEX idx_precio ON multimedia (precio) USING BTREE;
CREATE INDEX idx_vec ON multimedia (embedding) USING IVF WITH (lists = 100, metric = 'COSINE');

SET ivf.probes = 5;                 -- celdas que escanea el IVF
SET hybrid.strategy = 'AUTO';       -- AUTO | PRE | POST (forzar, para experimentos)
SET vector.scan = 'AUTO';           -- AUTO | EXACT (ignora el IVF: k-NN exacto, misma métrica)

SELECT id, categoria, precio, embedding <=> '[0.12, -0.34, ...]' AS dist_coseno
FROM multimedia
WHERE categoria = 'footwear' AND precio <= 250.0
ORDER BY dist_coseno ASC LIMIT 5;
```

`<->` es distancia L2 y `<=>` distancia coseno (1 − cos θ). El `WHERE` ahora admite
varios predicados sobre columnas distintas: el planificador resuelve uno con la mejor
ruta de acceso y evalúa el resto como filtro residual. También hay `ORDER BY <col>`.

### Rutas de acceso vectoriales

| Situación | Ruta | Qué hace |
|---|---|---|
| sin `WHERE`, sin IVF | `KnnSeqScan` | k-NN exacto, Max-Heap de tamaño k |
| sin `WHERE`, con IVF | `IVFScan` | escanea las `ivf.probes` celdas más cercanas |
| `WHERE` indexable, sin IVF | `PreFilter` | B+/Hash/clave entrega candidatos → k-NN exacto |
| `WHERE` no indexable, con IVF | `PostFilter` | candidatos IVF en orden → filtro; amplía el sondeo si no alcanza |
| `WHERE` indexable y con IVF | `PreFilter` / `PostFilter` | decide `hybrid.choose_strategy` con un conteo real del índice |
| `MATCH` + distancia | `TextPreFilter` | el índice invertido filtra → k-NN exacto sobre los candidatos |

`hybrid.choose_strategy` estima los bloques de cada estrategia: PRE ≈ filas que cumplen
el filtro (una lectura de tabla por fila); POST ≈ páginas de las celdas necesarias para
revisar k/selectividad candidatos + una lectura de tabla por candidato. Gana la menor;
ante un empate, PRE (es exacto).

Cada consulta reporta en `plan.io` las lecturas **del índice** y **de la tabla** por
separado (enunciado 5.4). El panel de métricas del cliente web las dibuja.

### Diseño en disco del IVF

| Archivo | Registro | Metadatos |
|---|---|---|
| `<t>__<idx>.ivf.dir` | `⟨centroide VECTOR(D), cluster_id, head, tail, count⟩` | `user_a` = métrica, `user_b` = C |
| `<t>__<idx>.ivf.lst` | `⟨pk, page, slot, area, vec⟩`, cadena de páginas por celda | `user_a` = N |

- C = ⌊√N⌋ por defecto (K-Means++ / Lloyd; esférico con `COSINE`).
- Las listas son densas: una página se decodifica de una vez con `numpy.frombuffer`.
- Se guarda la **PK junto al RID** porque en este motor los RID no son estables
  (move-the-last del Heap, inserción ordenada y `reorganize()` del Sequential). Al leer
  un candidato se verifica la PK; si el RID quedó obsoleto, se recupera por clave
  (`plan.stale_rids`). El `DELETE` sobre Heap además reubica la fila movida en el IVF.
- Los centroides viven en RAM después de abrir el índice (se leen una vez).

### Experimento 8

```bash
python -m backend.benchmarks.experiment8                  # sintético, N=20k, D=128
python -m backend.benchmarks.experiment8 --npy emb.npy    # embeddings reales (N, D)
python -m backend.benchmarks.experiment8 --csv data/datasets/wiki_20k.csv --metric COSINE
#   embeddings SBERT del corpus del RAG (se calculan una vez en <csv>.sbert.npy);
#   las consultas son documentos reales que se dejan fuera de la tabla
```

Escribe `benchmarks/results/exp8_ann.csv` (Recall@10 / ms / bloques vs `probes`, con el
k-NN exacto como verdad de campo) y `exp8_hibrido.csv` (PRE vs POST según selectividad);
con `--csv`, `exp8_ann_sbert.csv` y `exp8_hibrido_sbert.csv`.

## Aplicación de IA y módulo multimedia (Rol 4)

Enunciado 5.3.1 (BoVW vs. embeddings densos), 5.5 opción A (RAG) y Experimento 7.
Código en `backend/multimedia/`, `backend/rag/`, `backend/benchmarks/experiment7.py`
y los paneles `RagPanel.jsx` / `MultimediaPanel.jsx` del cliente web.

### Datos

```bash
# corpus del RAG: 7 500 artículos de Wikipedia por idioma (es + en = 15 000),
# API de filas de Hugging Face, reanudable y con espera ante HTTP 429
python -m backend.rag.download --n 7500 --out data/datasets/wiki_15k.csv

# imágenes: Caltech-101 (9 145 imágenes, 101 categorías) desde data.caltech.edu
python -m backend.multimedia.datasets --root data/datasets
```

### Construir el corpus y la tienda de imágenes

Desde la interfaz (botones **Construir corpus** / **Construir tienda**, con progreso) o
por consola:

```bash
python -m backend.rag.corpus --csv data/datasets/wiki_15k.csv        # SBERT + tabla + IVF + índice invertido
python -m backend.multimedia.service --k 256                          # SIFT + K-Means + ResNet-18 + IVF + índice invertido
```

Lo que queda en disco, todo dentro del motor:

| Objeto | Dónde | Para qué |
|---|---|---|
| `documentos (id, lang, titulo CHAR(80), texto CHAR(1000), emb VECTOR(384))` | `data/documentos.heap.dat` | fragmentos que cita el RAG |
| `documentos__idx_emb.ivf.*` | `data/` | recuperación semántica (SBERT + IVF COSINE) |
| `documentos__texto.vocab.dic` + `.postings.inv` + `.norms.dat` | `data/text/` | recuperación léxica (TF-IDF coseno y BM25 sobre el SPIMI de la Parte 3) |
| `imagenes (id, categoria, archivo, emb VECTOR(512))` | `data/imagenes.heap.dat` | ResNet-18 de cada imagen |
| `imagenes__idx_emb.ivf.*` | `data/` | paradigma moderno (IVF COSINE) |
| `codebook_bovw (id, centro VECTOR(128))` | `data/codebook_bovw.heap.dat` | K palabras visuales (K-Means sobre SIFT) |
| `imagenes__bovw.spimi.*` + `.norms.dat` | `data/text/` | paradigma clásico: histograma BoVW indexado en el índice invertido |

Los índices textuales viven en `data/text/` porque el catálogo descubre los índices
relacionales por el nombre del archivo en `data/` y el vocabulario (un B+ de
términos) no debe aparecer como índice de la tabla. `data/cache/` guarda las
características ya extraídas (`.npy`) para repetir el Experimento 7 sin volver a pagar
la CNN; no es almacenamiento del gestor.

### Modelo de lenguaje

El proveedor se elige por variables de entorno; sin ninguna, el RAG sigue funcionando
en modo **extractivo** (recupera, arma el prompt y cita, pero no genera texto):

| Variable | Proveedor | Modelo por defecto |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic (SDK oficial) | `claude-opus-5` |
| `GEMINI_API_KEY` | Google (endpoint compatible OpenAI) | `gemini-2.5-flash` |
| `GROQ_API_KEY` / `OPENAI_API_KEY` | Groq / OpenAI | `llama-3.3-70b-versatile` / `gpt-4o-mini` |
| `LLM_BASE_URL` + `LLM_API_KEY` | cualquier API `/chat/completions` (Ollama, OpenRouter…) | `LLM_MODEL` |
| `LLM_PROVIDER=local` | `transformers` en CPU | `Qwen/Qwen2.5-1.5B-Instruct` |

`LLM_PROVIDER` fuerza uno y `LLM_MODEL` cambia el modelo.

### API

| Método | Ruta | Qué hace |
|---|---|---|
| `GET` | `/api/rag/status` | corpus, tamaños de los índices y proveedor LLM |
| `POST` | `/api/rag/build` | `{limit?, lists?}` construye el corpus en segundo plano |
| `POST` | `/api/rag/ask` | `{question, k, mode: semantic/lexical/both, probes, scorer: tfidf/bm25, generate}` |
| `GET` | `/api/multimedia/status` | tienda de imágenes |
| `POST` | `/api/multimedia/build` | `{k_words, max_per_category?, lists?}` en segundo plano |
| `POST` | `/api/multimedia/search` | multipart `file` o `image_id` + `k` + `probes` → top-k por ambos paradigmas |
| `GET` | `/api/multimedia/samples` · `/image/{id}` | imágenes de consulta y archivos |

Cada respuesta trae, por ruta, la ruta de acceso del planificador (`InvertedIndex`,
`IVFScan`, `KnnSeqScan`), la latencia y los bloques leídos del **índice** y de la
**tabla** por separado.

### Experimento 7

```bash
python -m backend.benchmarks.experiment7                     # imágenes: BoVW K=128/256/512 vs ResNet exacto/IVF
python -m backend.benchmarks.experiment7 --text --text-n 4000  # además 7b: TF-IDF/BM25 vs SBERT en Wikipedia
```

Escribe `benchmarks/results/exp7_imagenes.csv` (Precision@5/@10, ms de extracción,
latencia y bloques por método) y `exp7_texto.csv` (hit@1, hit@10, MRR).

### Informe del Entregable 2

`informe_analisis/informe_p2.tex` es el documento maestro; cada rol edita su archivo en
`informe_analisis/secciones/` (`espacial`, `textual`, `vectorial`, `multimedia`). Los
números salen de los CSV:

```bash
python informe_analisis/figuras/generar_p2.py          # figuras de los Experimentos 7 y 8
python informe_analisis/secciones/exp7_tabla.py        # tablas del Experimento 7
python informe_analisis/secciones/sincronizar_espacial.py  # copia el cuerpo de experimento5_rtree.tex
```

Se compila con `pdflatex informe_p2.tex` desde `informe_analisis/` (o subiendo la
carpeta a Overleaf). El guion del video está en `tutorial_video_demostrativo_p2.md`.

### Módulo espacial (R-Tree)

El R-Tree se persiste en `<tabla>__<indice>.rtree.dat` sobre la misma cabecera de
página de 32 B. Cada entrada es `MBR (2·D float32) + RID (page_id, slot, area)`;
`root_page_id` y `height` viven en `user_a`/`user_b`.

```sql
CREATE TABLE locales (id INT PRIMARY KEY, nombre CHAR(20), ubicacion POINT) USING HEAP;
INSERT INTO locales VALUES (1, 'Miraflores', POINT(-77.0282, -12.1354));
CREATE INDEX idx_geo ON locales (ubicacion) USING RTREE;

SELECT id, ubicacion FROM locales
WHERE ubicacion WITHIN_RADIUS ((-77.0282, -12.1354), 2.5);

SELECT id, ubicacion FROM locales
WHERE ubicacion KNN ((-77.0282, -12.1354), 10);
```

- `WITHIN_RADIUS` usa la caja conservadora del radio y refina con haversine.
- `KNN` usa una cola de prioridad por `MINDIST(Q, MBR)` y poda con la distancia
  del k-ésimo vecino actual.
- Sin índice el planificador cae a `SpatialSeqScan`.
- En tablas `SEQUENTIAL` la inserción ordenada mueve slots, así que cada
  `INSERT` reindexa el R-Tree para mantener los RID válidos. Crea el índice
  después de la carga masiva, o usa `HEAP` si vas a insertar después.
- El visor web muestra los puntos, el círculo/centro de consulta y los MBRs
  inspeccionados (`visited_mbrs`) en la pestaña **Mapa espacial**.

### Experimento 5

```bash
python -m backend.benchmarks.experiment5
```

Escribe `benchmarks/results/exp5_espacial.csv` (R-Tree vs escaneo secuencial para
radio y k-NN con N = 10k, 50k, 100k) y `exp5_dimension.csv` (maldición de la
dimensionalidad para D = 2…128). Las figuras y el informe están en
`informe_analisis/experimento5_rtree.tex` (requiere `matplotlib` para
`informe_analisis/figuras/generar.py`).

### Módulo textual en SQL (índice invertido)

Enunciado 5.2 y 5.4. Código en `backend/text/` (`boolean.py`, `inverted.py`, SPIMI) y
`Executor._select_text`.

- `CREATE INDEX ... USING INVERTED WITH (scoring = 'TFIDF_COSINE' | 'BM25')` sobre una
  columna `CHAR`. Archivos: `<t>__<idx>.inv.meta` (columna, scoring, si está
  desactualizado), diccionario `vocab.dic` (B+ paginado término → offset, df; el largo
  de la lista es df), listas de posteo `postings.inv` ordenadas por RID y normas
  (`.inv.norms.dat`).
- `MATCH` con operadores (`AND`, `OR`, `NOT`, paréntesis; en mayúsculas) es booleano y
  se resuelve con mezclas de dos punteros en O(n + m); sin operadores es una consulta de
  ranking (OR de las palabras). `COSINE_SCORE()` / `BM25_SCORE()` eligen el score.
- Ruta `InvertedIndexScan` (filtro y refinamiento): el índice da los documentos y su
  score sin tocar la tabla; después se leen las filas por RID en orden de relevancia,
  se aplican los demás predicados y con `LIMIT` se para al juntar k. Sin índice:
  `TextSeqScan` (mismo resultado, lee toda la tabla).
- Un `INSERT`, `DELETE` o reorganización marca el índice como desactualizado y el
  siguiente `MATCH` lo reconstruye (el plan lo indica).

El RAG y el BoVW usan los mismos archivos SPIMI: `build_index(..., tokenize=)` acepta un
tokenizador alternativo (las palabras visuales no pasan por stopwords ni stemming).

### Experimento 6

```bash
docker run -d --name bd2-pg -e POSTGRES_PASSWORD=bd2 -e POSTGRES_DB=bd2 -p 55432:5432 postgres:16-alpine
python -m backend.benchmarks.experiment6 --csv data/datasets/wiki_20k.csv   # corpus: python -m backend.rag.download --n 10000 --out data/datasets/wiki_20k.csv
python -m backend.benchmarks.experiment6 --no-pg                            # sin PostgreSQL
```

Escribe `exp6_construccion.csv` (tiempo y tamaño del índice vs GIN, 1k → 20k documentos)
y `exp6_consultas.csv` (AND, OR, AND-NOT y Top-10 de 1, 3 y 5 términos: índice vs Full
Text Scan vs GIN). PostgreSQL es solo la referencia externa del enunciado (necesita
`psycopg2`); el motor no lo usa.

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


## Limitaciones conocidas

- `SequentialFile` exige clave primaria `INT` o `FLOAT`: el `low_key` de la cabecera de
  página ocupa 8 bytes. Una PK `CHAR(n)` levanta `UnsupportedKeyType`.
- `reorganize()` invalida los `RID` previamente entregados, igual que cualquier
  reorganización de archivo.
