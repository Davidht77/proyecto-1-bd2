# `data/` — archivos binarios del motor y datasets

Esta carpeta se versiona vacía (solo este README). Todo lo que aparece aquí lo genera
el motor o los scripts de descarga; nada se edita a mano. La ruta se cambia con
`BD2_DATA_DIR`.

| Ruta | Qué es | Quién la genera |
|---|---|---|
| `<tabla>.heap.dat`, `<tabla>.seq.dat` + `.seq.ovf` | tablas (Heap File / Sequential File con overflow) | `CREATE TABLE ... USING HEAP \| SEQUENTIAL` |
| `<tabla>__<idx>.btree.dat` | Árbol B+ | `CREATE INDEX ... USING BTREE` |
| `<tabla>__<idx>.hash.dir` + `.hash.bkt` | Extendible Hashing | `CREATE INDEX ... USING HASH` |
| `<tabla>__<idx>.rtree.dat` | R-Tree | `CREATE INDEX ... USING RTREE` |
| `<tabla>__<idx>.ivf.dir` + `.ivf.lst` | IVF: directorio de centroides + listas invertidas paginadas | `CREATE INDEX ... USING IVF` |
| `<tabla>__<idx>.inv.meta`, `.inv.vocab.dic`, `.inv.postings.inv`, `.inv.norms.dat` | índice invertido (diccionario, listas de posteo y normas L2) | `CREATE INDEX ... USING INVERTED` |
| `text/` | índices invertidos del RAG (`documentos__texto.*`) y del BoVW (`imagenes__bovw.*`) | `python -m backend.rag.corpus`, `POST /api/multimedia/build` |
| `datasets/wiki_15k.csv`, `datasets/wiki_20k.csv` | corpus de Wikipedia (es + en) | `python -m backend.rag.download` |
| `datasets/caltech101/` | imágenes de Caltech-101 | `POST /api/multimedia/build` |
| `cache/` | características ya extraídas (`.npy`) para no recalcular la CNN | módulo multimedia |
| `benchmarks/` | archivos temporales de los experimentos | `backend/benchmarks/*` |

Los resultados de los experimentos van a `benchmarks/results/` (CSV versionados).
