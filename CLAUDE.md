# CLAUDE.md — proyecto-1-bd2

Reglas de este repo. Tienen prioridad sobre el CLAUDE.md global del usuario.

## Git

- Prohibido `git push`. Solo commits locales.
- Prohibido añadir a Claude como autor o coautor de un commit (sin línea `Co-Authored-By: Claude`, sin cambiar `git config user.*`).

## Hashing Dinámico (backend/index/hash.py)

Implementación: **Extendible Hashing**, siguiendo el PPT "05 Hash, Bitmap Index
Scan y BRIN en PostgreSQL" (Semana 05, CS2042, diapositivas 16–27). Fuente:
`~/Descargas/05 Hash, Bitmap Index Scan y BRIN en PostgreSQL.pdf`.

Algoritmo (según el PPT):

- Directorio de `2^D` punteros a buckets, `D` = profundidad global. Cada
  bucket tiene profundidad local `d ≤ D`.
- Índice de bucket = últimos `D` bits de `hash(key)`.
- Búsqueda: aplicar hash, tomar el bucket del directorio, recorrer sus
  entradas.
- Inserción: si el bucket tiene espacio, insertar. Si está lleno, dividirlo
  (`d = d + 1`, reinsertar sus registros, actualizar el directorio); si
  `d == D`, duplicar el directorio primero (`D = D + 1`).
- Eliminación: quitar la entrada; si el bucket y su buddy (mismo prefijo en
  `d - 1`) caben juntos, fusionarlos y liberar la página sobrante.

Mapeo sobre la capa de storage (`backend/storage/pager.py`,
`backend/storage/page.py`, ya construida por el resto del equipo):

- `pager.user_a` = profundidad global `D` (en el pager del **directorio**).
- `page.aux_page_id` = profundidad local del bucket.
- `pager.free_page(id)` libera buckets vacíos tras una fusión.

Dos archivos físicos por índice (mismo patrón que `.seq.dat`/`.seq.ovf`):
`<tabla>__<indice>.hash.dir` (directorio) y `<tabla>__<indice>.hash.bkt`
(buckets).

## Prioridades actuales

Los 4 experimentos del informe (`POST /api/benchmarks/run`, carpetas `data/`
y `benchmarks/`) quedan para después de cerrar el Hashing Dinámico. No
priorizar ese endpoint mientras el índice esté en construcción.
