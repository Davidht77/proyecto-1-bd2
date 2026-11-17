"""Regenera secciones/espacial.tex a partir de experimento5_rtree.tex (Rol 1).

El documento del Rol 1 es autocontenido (tiene su propio preambulo). Este script
toma solo el cuerpo, baja un nivel los encabezados y lo deja como seccion del
informe maestro informe_p2.tex, para no mantener dos copias a mano.

    python informe_analisis/secciones/sincronizar_espacial.py
"""

import pathlib

aqui = pathlib.Path(__file__).resolve().parent
src = (aqui.parent / "experimento5_rtree.tex").read_text(encoding="utf-8")
cuerpo = src.split("\\maketitle", 1)[1].split("\\end{document}", 1)[0]
cuerpo = cuerpo.replace("\\subsection{", "\\subsubsection{").replace("\\section{", "\\subsection{")
cabecera = (
    "\\section{Módulo espacial: R-Tree en disco}\n\\label{sec:espacial}\n\n"
    "% Cuerpo tomado de experimento5_rtree.tex (Rol 1) por un script; si editas\n"
    "% el original, vuelve a correr: python informe_analisis/secciones/sincronizar_espacial.py\n"
)
(aqui / "espacial.tex").write_text(cabecera + cuerpo, encoding="utf-8")
print("espacial.tex regenerado:", len(cuerpo.splitlines()), "lineas")
