# Guion del video demostrativo — Proyecto 2 (5 a 10 minutos)

El enunciado (sección 7) pide que **todos los integrantes** inspeccionen archivos
binarios en disco, ejecuten consultas SQL y la aplicación de IA en el cliente web
mirando el plan de ejecución y las lecturas de disco, y expongan las gráficas del
benchmark. Este guion reparte esos momentos por rol y cabe en 9 minutos. El guion del
Proyecto 1 sigue en `tutorial_video_demostrativo.md`.

## 0. Preparación (antes de grabar)

```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
# datasets (una sola vez)
python -m backend.rag.download --n 7500 --out data/datasets/wiki_15k.csv
python -m backend.multimedia.datasets --root data/datasets
# corpus e índices (minutos; o desde los botones "Construir" de la interfaz)
python -m backend.rag.corpus --csv data/datasets/wiki_15k.csv
python -m backend.multimedia.service --k 256
# opcional: modelo de lenguaje
set GEMINI_API_KEY=...        # o ANTHROPIC_API_KEY
python -m uvicorn backend.api.main:app --port 8000
```

Abrir <http://127.0.0.1:8000>. Tener listos: un editor hexadecimal (HxD o la extensión
Hex Editor de VS Code), la carpeta `data/`, y las figuras de `informe_analisis/figuras/`.

## 1. Apertura (30 s) — quien presenta

> "Es el mismo motor del Proyecto 1: páginas de 4 KB, RID ⟨page, slot⟩ y un contador de
> I/O. Hoy mostramos cómo sobre esas mismas páginas viven un R-Tree, un índice
> invertido, un índice vectorial IVF y, encima de todo, una aplicación de IA."

Mostrar la barra superior: **Cliente SQL · Aplicación IA · Multimedia**.

## 2. Módulo espacial (1 min 30 s) — Rol 1

1. Cliente SQL: botón de ejemplo **crear tabla espacial**, **insertar punto**, sembrar
   10 000 puntos (botón de carga), `CREATE INDEX idx_geo ... USING RTREE`.
2. Ejecutar **radio** y **k-NN espacial**; abrir la pestaña **Mapa espacial**: puntos,
   círculo de consulta y MBRs visitados.
3. Panel de métricas: ruta `RTree...` frente a `SpatialSeqScan`, bloques leídos.
4. Hexadecimal: abrir `data/locales__idx_geo.rtree.dat`; señalar cabecera de página y
   una entrada MBR (4 float32) + RID.
5. Figura `exp5_radio.pdf` / `exp5_dimension.pdf`: "a partir de D≈16 el solapamiento
   hace que el árbol lea casi todas las hojas".

## 3. Módulo textual (1 min 30 s) — Rol 2

1. Mostrar `data/text/documentos__texto.vocab.dic` y
   `...postings.inv` en el hexadecimal: diccionario (B+ de términos de 24 bytes) separado
   de las listas de posteo (registros ⟨page, slot, tf, dl⟩ de 16 bytes, 252 por página).
2. Consulta booleana y ranking top-k en el Cliente SQL: `CREATE INDEX ... USING INVERTED`,
   `WHERE texto MATCH 'a AND b AND NOT c'` con `COSINE_SCORE()` / `BM25_SCORE()`; el plan
   muestra `InvertedIndexScan` frente a `TextSeqScan`. También desde la vista
   **Aplicación IA** con la ruta léxica.
3. Figura del Experimento 6: construcción y tamaño vs. número de documentos; latencia
   y bloques frente a full scan y GIN de PostgreSQL.

## 4. Módulo vectorial y SQL híbrido (1 min 30 s) — Rol 3

1. Cliente SQL sobre `documentos` o `imagenes`: `SET ivf.probes = 1`, luego `5`,
   misma consulta `ORDER BY emb <=> '[…]' LIMIT 5`; mostrar en el panel cómo cambian
   *celdas sondeadas* y *lecturas índice / tabla*.
2. Consulta híbrida (`WHERE lang = 'es' ... ORDER BY ... LIMIT 5`) y la *Decisión del
   planificador* (PRE vs POST) con `SET hybrid.strategy`.
3. Hexadecimal: `data/documentos__idx_emb.ivf.dir` (centroides) y `.ivf.lst`
   (listas paginadas ⟨pk, page, slot, area, vec⟩).
4. Figuras `exp8_recall.pdf` y `exp8_hibrido.pdf`: la rodilla en probes = 5 y el
   cruce PRE/POST entre 1 % y 5 % de selectividad.

## 5. Multimedia y aplicación de IA (2 min 30 s) — Rol 4

### 5.1 Multimedia (1 min)
1. Vista **Multimedia**: estado (8 677 imágenes, 256 palabras visuales, celdas IVF).
2. Elegir una imagen de la muestra (p. ej. una motocicleta): dos columnas, **BoVW**
   (SIFT → codebook → índice invertido con TF-IDF coseno) y **ResNet-18 + IVF**.
   Señalar: borde verde/rojo por categoría, *Precision@10* de cada ruta, ms de
   extracción (SIFT vs CNN), bloques leídos índice/tabla, ruta `InvertedIndex` vs
   `IVFScan`.
3. Subir una foto propia desde el disco y repetir.
4. Mencionar dónde vive cada cosa: tabla `imagenes` con `VECTOR(512)`, tabla
   `codebook_bovw` con `VECTOR(128)`, `imagenes__bovw.spimi.*` + `.norms.dat`.

### 5.2 RAG (1 min 30 s)
1. Vista **Aplicación IA · RAG**: estado (15 000 documentos, tamaños de vocabulario y
   postings, celdas IVF, proveedor LLM).
2. Pregunta de ejemplo en español; contexto = *ambos*. Mostrar:
   - columna **Léxico** (índice invertido, TF-IDF coseno; cambiar a BM25),
   - columna **Semántico** (`ORDER BY emb <=> … LIMIT k`, `IVFScan`, probes),
   - métricas de cada ruta: ms, lecturas de índice vs tabla,
   - fragmentos resaltados que aparecen en ambas rutas,
   - la **respuesta generada con citas [n]** enlazadas a las fuentes (o la respuesta
     extractiva si no hay clave de LLM, explicando que el pipeline es el mismo).
3. Repetir con una pregunta en inglés y `probes = 1` para ver caer las lecturas.
4. Hexadecimal: `data/documentos.heap.dat` (registro con `CHAR(1000)` + 384 float32).

### 5.3 Experimento 7 (30 s)
Figuras `exp7_precision.pdf` y `exp7_costos.pdf`: "el embedding denso gana en
Precision@k; SIFT es más barato de extraer y el índice invertido lee menos páginas
por consulta". Si se corrió 7b, `exp7_texto.pdf`: léxico vs SBERT en ítem conocido.

## 6. Cierre (30 s)

> "Cuatro módulos, un solo Pager, un solo contador de I/O: por eso cada gráfica del
> informe compara bloques físicos reales, y por eso la aplicación de IA no necesitó
> ninguna estructura fuera del motor."

## Lista de verificación de la rúbrica

- [ ] Los cuatro integrantes aparecen y cada uno inspecciona al menos un binario.
- [ ] Consultas SQL multimodales con plan de ejecución y lecturas/escrituras visibles.
- [ ] Aplicación de IA funcionando en el cliente web con métricas de I/O.
- [ ] Gráficas de los Experimentos 5, 6, 7 y 8 explicadas a la luz de la teoría.
- [ ] Duración entre 5 y 10 minutos.
