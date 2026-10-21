# -*- coding: utf-8 -*-
"""
run_notebook_v2.py
==================

Pipeline de construcción y ejecución del notebook final.

Pasos
-----
v0  Se construye el notebook con los marcadores `{{clave}}` intactos, se ejecuta
    con nbclient (cwd = carpeta de entrega, para que resuelvan los CSV) y se
    cosecha de las salidas todos los valores `@@clave=valor@@` publicados por las
    celdas de código.
v1  Se vuelve a construir el notebook sustituyendo cada marcador por el valor
    realmente calculado, se ejecuta de nuevo y ese es el notebook entregado:
    se eliminan las líneas `@@...@@` de las salidas y se suprime la celda
    técnica REPORTE.

El pipeline se ejecuta dos veces para comprobar que los valores cosechados son
idénticos (determinismo).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import build_notebook_v2 as B  # noqa: E402

CARPETA = BASE.parent / "Grupo_36_version2"
DESTINO = CARPETA / "Grupo_36_Taller3.ipynb"
RESULTADOS = BASE / "resultados.json"

MARCADOR = re.compile(r"@@([A-Za-z0-9_]+)=(.*?)@@")
LINEA_MARCA = re.compile(r"^@@[^@\n]*@@\s*$", re.MULTILINE)


# --------------------------------------------------------------------------
def ejecutar(nb, timeout: int = 3600):
    """Ejecuta el notebook en un kernel real, con cwd = carpeta de entrega."""
    cliente = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(CARPETA)}},
        allow_errors=False,
        log=__import__("logging").getLogger("nbclient"),
    )
    cliente.execute()
    return nb


def cosechar(nb) -> dict[str, str]:
    """Extrae los pares @@clave=valor@@ de todas las salidas de stream."""
    vals: dict[str, str] = {}
    vistos: dict[str, set[str]] = {}
    for celda in nb.cells:
        for salida in celda.get("outputs", []):
            if salida.get("output_type") != "stream":
                continue
            texto = salida.get("text", "")
            if isinstance(texto, list):
                texto = "".join(texto)
            for clave, valor in MARCADOR.findall(texto):
                vals[clave] = valor
                vistos.setdefault(clave, set()).add(valor)
    inconsistentes = {k: sorted(v) for k, v in vistos.items() if len(v) > 1}
    if inconsistentes:
        raise SystemExit(
            "Un mismo valor fue publicado con texto distinto en celdas diferentes:\n"
            + "\n".join(f"  {k}: {v}" for k, v in sorted(inconsistentes.items()))
        )
    return vals


def limpiar_salidas(nb) -> int:
    """Quita las líneas @@...@@ de las salidas de stream y borra las vacías."""
    quitadas = 0
    for celda in nb.cells:
        salidas = celda.get("outputs", [])
        nuevas = []
        for salida in salidas:
            if salida.get("output_type") == "stream":
                texto = salida.get("text", "")
                if isinstance(texto, list):
                    texto = "".join(texto)
                limpio, n = LINEA_MARCA.subn("", texto)
                quitadas += n
                if limpio.strip() == "":
                    continue          # salida vacía -> se elimina
                salida["text"] = limpio
            nuevas.append(salida)
        celda["outputs"] = nuevas
    return quitadas


def quitar_celda_reporte(nb) -> None:
    """Elimina la última celda (técnica REPORTE)."""
    if nb.cells and "@@" in nb.cells[-1].source and "Celda técnica de recolección" in nb.cells[-1].source:
        del nb.cells[-1]
    else:
        raise SystemExit("No se encontró la celda REPORTE al final del notebook")


def pipeline() -> dict[str, str]:
    """v0 -> cosecha -> v1 -> ejecución -> limpieza -> guardado."""
    print("=" * 78)
    print("PASO v0: construcción con marcadores + ejecución + cosecha")
    print("=" * 78)
    nb0 = B.construir(None)
    print(f"  notebook v0: {len(nb0.cells)} celdas")
    ejecutar(nb0)
    vals = cosechar(nb0)
    print(f"  valores cosechados: {len(vals)}")
    if not vals:
        raise SystemExit("No se cosechó ningún valor @@clave=valor@@")

    print("=" * 78)
    print("PASO v1: construcción renderizada + ejecución final")
    print("=" * 78)
    nb1 = B.construir(vals)
    ejecutar(nb1)
    quitar_celda_reporte(nb1)
    n = limpiar_salidas(nb1)
    print(f"  líneas @@ eliminadas de las salidas: {n}")

    for celda in nb1.cells:
        if celda.cell_type == "markdown" and ("{{" in celda.source or "}}" in celda.source):
            raise SystemExit("Quedó un marcador sin renderizar en el notebook final")

    nbformat.write(nb1, str(DESTINO))
    print(f"  notebook final escrito en: {DESTINO}")

    RESULTADOS.write_text(json.dumps(vals, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  resultados escritos en: {RESULTADOS}")
    return vals


def main() -> None:
    print("### EJECUCIÓN 1 ###")
    vals1 = pipeline()

    print()
    print("### EJECUCIÓN 2 (control de determinismo) ###")
    vals2 = pipeline()

    print()
    print("=" * 78)
    if vals1 == vals2:
        print(f"DETERMINISMO: OK - las dos ejecuciones produjeron {len(vals1)} valores idénticos")
    else:
        solo1 = {k: vals1[k] for k in vals1 if vals1.get(k) != vals2.get(k)}
        print("DETERMINISMO: FALLO - valores distintos entre ejecuciones:")
        for k, v in solo1.items():
            print(f"  {k}: {v!r}  !=  {vals2.get(k)!r}")
        raise SystemExit(1)

    print()
    print("=" * 78)
    print("VALORES FINALMENTE COLECTADOS")
    print("=" * 78)
    for k in sorted(vals1):
        print(f"{k} = {vals1[k]}")


if __name__ == "__main__":
    main()
