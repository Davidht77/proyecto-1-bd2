# Tutorial para el Video Demostrativo (Proyecto 1)

Este documento es una guía estructurada para grabar el video demostrativo requerido (de 5 a 10 minutos). Sigue estos pasos para cubrir todo lo que evalúa la rúbrica del Entregable 1.

## 1. Preparación del Sistema

Antes de grabar, asegúrate de levantar tanto el backend como el frontend, asegurando que el `BD2_CACHE_SIZE` esté en `0` si vas a mostrar métricas directas a disco.

```bash
# Terminal 1: Backend
python -m uvicorn backend.api.main:app --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

Abre `http://localhost:5173` (o el puerto que te asigne Vite) en el navegador.

---

## 2. Ejecución de Consultas y Planes de Ejecución (Cliente Web)

**Objetivo:** Demostrar el funcionamiento del Parser SQL, el Planificador de Consultas y el Motor (Heap, Sequential, B+, Hash).

### Paso 2.1: Creación de la tabla base
En el editor SQL del frontend ejecuta:
```sql
CREATE TABLE empleados (id INT PRIMARY KEY, nombre CHAR(30), salario FLOAT) USING SEQUENTIAL;
```
> **Narración sugerida:** "Aquí creamos una tabla `empleados` utilizando el motor de almacenamiento Sequential File. Vemos que se ha creado exitosamente el archivo `.seq.dat`".

### Paso 2.2: Carga masiva de datos (Seed)
En la interfaz del cliente web, busca el botón de **Carga de Datos (Seed)** (o hazlo vía API) para cargar unos 100,000 registros sintéticos.
> **Narración sugerida:** "Para las pruebas cargamos 100 mil registros. El Sequential file usa overflow y reorganización periódica automática".

### Paso 2.3: Índices y Rutas de Acceso
Ejecuta las siguientes consultas una por una mostrando la pestaña de **Plan de Ejecución y Métricas**:

1. **Búsqueda por clave (Binary Search en Sequential File)**
   ```sql
   SELECT * FROM empleados WHERE id = 50000;
   ```
   > **Mostrar:** Que la ruta de acceso (Access Path) es `BinarySearch` (o `IndexScan` si usas un índice secundario) y el costo es $O(\log N)$ páginas leídas.

2. **Búsqueda por rango**
   ```sql
   SELECT * FROM empleados WHERE id >= 1000 AND id <= 5000;
   ```
   > **Mostrar:** Que usa `BinaryRangeScan` o `IndexRangeScan`, logrando un recorrido óptimo sin leer toda la tabla.

3. **Búsqueda por columna sin indexar (SeqScan)**
   ```sql
   SELECT * FROM empleados WHERE nombre = 'Ada Lovelace';
   ```
   > **Mostrar:** Que el motor cae de nuevo en `SeqScan` (Full Table Scan) porque no hay índice, leyendo el 100% de los bloques de la tabla.

---

## 3. Inspección de Archivos Binarios en Disco

**Objetivo:** Demostrar el diseño físico (`Page Layout`, `RID`, Bloques de tamaño fijo).

### Opción A: A través del Frontend (Visualizador Físico)
El frontend incluye un "Mapa de Páginas Físicas". Ábrelo para la tabla `empleados`.
> **Narración sugerida:** "Aquí vemos las páginas de disco reales. Cada bloque (ej. de 4KB) muestra su `page_id`, la cantidad de registros (`record_count`), el factor de ocupación y los enlaces a su cadena de overflow (`next_page_id`). No estamos usando un ORM, estamos leyendo los metadatos directamente del binario".

### Opción B: A través de un Editor Hexadecimal (Opcional pero muy visual)
1. Abre un editor hexadecimal como *VSCode Hex Editor* o *HxD*.
2. Abre el archivo `data/empleados.seq.dat`.
> **Narración sugerida:** "Este es nuestro archivo en crudo. Aquí al inicio (primeros 32 bytes) vemos la cabecera con el `page_id`, `record_count`, y el `free_space_offset`. El resto son nuestros registros empaquetados usando representaciones binarias enteras fijas y el bitmap".

---

## 4. Resultados del Benchmark Automatizado (Fase Experimental)

**Objetivo:** Presentar las métricas de rendimiento y comparativas de IO.

Dirígete a una herramienta de peticiones REST (como Thunder Client, Postman o el Swagger en `http://127.0.0.1:8000/docs`).

1. Llama al endpoint `POST /api/benchmarks/run`. 
   *(Nota: Puedes correr uno por uno con el body `{"experiment": 2}` si tarda mucho el conjunto completo).*
2. Recibirás un JSON masivo.

> **Narración sugerida:** "Finalmente ejecutamos la suite de Benchmarks. 
> - **Exp 1:** Muestra que las inserciones en Sequential File aumentan drásticamente su latencia cuando entra en el área de overflow y debe reorganizarse.
> - **Exp 2:** Las consultas puntuales demuestran que el B+ Tree y Hashing tienen lecturas $O(1)$ o $O(\log N)$, aplastando por completo al Full Scan.
> - **Exp 3 (Rangos):** El Árbol B+ es extremadamente eficiente para rangos frente al Hashing Dinámico y al Heap.
> - **Exp 4 (Bloques):** Variando $B$ vemos que a mayor tamaño de página (ej. 8192 bytes), el `fan-out` aumenta, reduciendo la altura final $h$ del árbol B+ y minimizando la latencia IO general."

---
¡Listo! Muestra seguridad al presentar, ya que todas las estructuras de persistencia las han creado desde 0 sobre memoria secundaria.
