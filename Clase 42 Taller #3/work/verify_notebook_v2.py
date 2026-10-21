# -*- coding: utf-8 -*-
"""
verify_notebook_v2.py
=====================

Comprobaciones automáticas sobre el notebook final entregado
(`Grupo_36_version2/Grupo_36_Taller3.ipynb`).

1. Estructura: cantidad de celdas, ausencia de la celda técnica REPORTE,
   ausencia de marcadores `{{...}}` y de residuos `@@...@@` en las salidas,
   ausencia de errores de ejecución, `execution_count` presente en todas las
   celdas de código y figuras realmente generadas.
2. Consistencia del render: el markdown del notebook entregado debe ser
   exactamente el que se obtiene renderizando las plantillas de
   `build_notebook_v2.CELLS` con los valores cosechados en `resultados.json`.
3. Coherencia numérica: más de 60 números clave del texto se contrastan con la
   salida de la celda de código que los produce, y se recalculan algunas
   magnitudes derivadas a partir de los propios valores cosechados.

Imprime un resumen PASS/FAIL y termina con código de salida 1 si algo falla.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nbformat

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import build_notebook_v2 as B  # noqa: E402

CARPETA = BASE.parent / "Grupo_36_version2"
NOTEBOOK = CARPETA / "Grupo_36_Taller3.ipynb"
RESULTADOS = BASE / "resultados.json"
PDFS = ["Apunte_Teoria_y_Practica_Taller3.pdf", "Resumen_codigo_Taller3.pdf"]

ESPERADOS = {"markdown": 37, "code": 15, "total": 52}
N_FIGURAS_ESPERADAS = 5

resultados: list[tuple[bool, str, str]] = []


def check(ok: bool, nombre: str, detalle: str = "") -> bool:
    resultados.append((bool(ok), nombre, detalle))
    return bool(ok)


def a_numero(texto: str) -> float:
    """'1,1080' -> 1.108 ; '71 188' -> 71188 ; 'p = 42' -> 42."""
    t = str(texto).replace("\u00a0", " ")
    t = re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "", t)      # separador de miles
    m = re.search(r"-?\d+(?:[.,]\d+)?", t)
    if m is None:
        raise ValueError(f"no hay número en {texto!r}")
    return float(m.group(0).replace(",", "."))


def texto_celda(celda) -> str:
    partes = []
    for salida in celda.get("outputs", []):
        if salida.get("output_type") == "stream":
            v = salida.get("text", "")
            partes.append("".join(v) if isinstance(v, list) else v)
        elif salida.get("output_type") == "execute_result":
            partes.append(str(salida.get("data", {}).get("text/plain", "")))
        elif salida.get("output_type") == "display_data":
            partes.append(str(salida.get("data", {}).get("text/plain", "")))
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Contraste prosa <-> salida.
# (descripción, celda_markdown, evidencia_markdown, celda_código,
#  evidencia_salida, número_en_la_prosa, número_en_la_salida,
#  tolerancia, factor_prosa, factor_salida)
# Los índices de celda son sobre la lista completa de celdas del notebook.
# ---------------------------------------------------------------------------
CHECKS = [
    # --- 2(b) comparación OOF de los modelos (md 26 / código 25) ---
    ("OOF log loss de Gradient Boosting", 26, "0,3111", 25, "0.3111", "0,3111", "0.3111", 1e-9, 1, 1),
    ("OOF Brier de Gradient Boosting", 26, "0,0896", 25, "0.0896", "0,0896", "0.0896", 1e-9, 1, 1),
    ("OOF AUC de Gradient Boosting", 26, "0,7442", 25, "0.7442", "0,7442", "0.7442", 1e-9, 1, 1),
    ("OOF log loss de Random Forest", 26, "0,3122", 25, "0.3122", "0,3122", "0.3122", 1e-9, 1, 1),
    ("OOF log loss de la logística con 42 columnas", 26, "0,3151", 25, "0.3151", "0,3151", "0.3151", 1e-9, 1, 1),
    ("OOF AUC de la logística con 42 columnas", 26, "0,7250", 25, "0.7250", "0,7250", "0.7250", 1e-9, 1, 1),
    ("OOF log loss de la logística con 10 columnas F", 26, "0,3203", 25, "0.3203", "0,3203", "0.3203", 1e-9, 1, 1),
    ("OOF log loss de la referencia", 26, "0,3574", 25, "0.3574", "0,3574", "0.3574", 1e-9, 1, 1),
    ("OOF log loss de Naive Bayes", 26, "2,2559", 25, "2.2559", "2,2559", "2.2559", 1e-9, 1, 1),
    ("OOF AUC del SVM radial", 26, "0,6851", 25, "0.6851", "0,6851", "0.6851", 1e-9, 1, 1),
    ("log loss medio de entrenamiento, árbol prof. 3", 26, "0,3199", 25, "0.3199", "0,3199", "0.3199", 1e-9, 1, 1),
    ("log loss medio de entrenamiento, árbol prof. 6", 26, "0,3019", 25, "0.3019", "0,3019", "0.3019", 1e-9, 1, 1),
    ("OOF AUC del árbol de profundidad 6", 29, "0,6636", 25, "0.6636", "0,6636", "0.6636", 1e-9, 1, 1),
    ("brecha entrenamiento-OOF del árbol de profundidad 6", 29, "0,0374", 28, "0.0374", "0,0374", "0.0374", 1e-9, 1, 1),
    # --- 2(c) SVM (md 29 / código 28) ---
    ("AUC del SVM sin estandarizar las numéricas", 29, "0,5986", 28, "0.5986", "0,5986", "0.5986", 1e-9, 1, 1),
    ("AUC del SVM con gamma = 0,1/p (peor de la grilla)", 29, "0,6279", 28, "0.6279", "0,6279", "0.6279", 1e-9, 1, 1),
    ("AUC del SVM con gamma = 10/p (mejor de la grilla)", 29, "0,7002", 28, "0.7002", "0,7002", "0.7002", 1e-9, 1, 1),
    # --- 3(a) umbrales y costos (md 33 / código 32) ---
    ("umbral elegido por costo mínimo", 33, "0,15", 32, "0.15", "0,15", "0.15", 1e-9, 1, 1),
    ("costo C(t = 0,15) sobre OOF", 33, "1492", 32, "1492", "1492", "1492", 1e-9, 1, 1),
    ("costo por registro C/n en t = 0,15", 33, "0,4125", 32, "0.412", "0,4125", "0.412", 1e-3, 1, 1),
    ("falsos positivos con t = 0,05", 33, "2356", 32, "2356", "2356", "2356", 1e-9, 1, 1),
    ("costo con t = 0,50", 33, "1816", 32, "1816", "1816", "1816", 1e-9, 1, 1),
    ("registros contactados con t = 0,15", 33, "559", 32, "559", "559", "559", 1e-9, 1, 1),
    ("sensibilidad con t = 0,15 (%)", 33, "46", 32, "0.460", "46", "0.460", 1e-9, 1, 100),
    ("especificidad con t = 0,15 (%)", 33, "88,5", 32, "0.885", "88,5", "0.885", 1e-9, 1, 100),
    ("verdaderos positivos con t = 0,15 sobre OOF", 33, "192", 32, "192", "192", "192", 1e-9, 1, 1),
    ("costo de no contactar a nadie", 33, "2085", 32, "2085", "2085", "2085", 1e-9, 1, 1),
    ("costo de contactar a todos", 33, "3200", 32, "3200", "3200", "3200", 1e-9, 1, 1),
    # --- 3(b) evaluación en prueba (md 36 / código 35) ---
    ("log loss del modelo en prueba", 36, "0,3087", 35, "0.3087", "0,3087", "0.3087", 1e-9, 1, 1),
    ("Brier del modelo en prueba", 36, "0,0886", 35, "0.0886", "0,0886", "0.0886", 1e-9, 1, 1),
    ("AUC del modelo en prueba", 36, "0,7504", 35, "0.7504", "0,7504", "0.7504", 1e-9, 1, 1),
    ("log loss de la referencia en prueba", 36, "0,3569", 35, "0.3569", "0,3569", "0.3569", 1e-9, 1, 1),
    ("C/n del modelo con t = 0,15 en prueba", 36, "0,3938", 35, "0.3938", "0,3938", "0.3938", 1e-9, 1, 1),
    ("C/n del modelo con t = 0,50 en prueba", 36, "0,4768", 35, "0.4768", "0,4768", "0.4768", 1e-9, 1, 1),
    ("C/n de la referencia en prueba", 36, "0,5752", 35, "0.5752", "0,5752", "0.5752", 1e-9, 1, 1),
    ("C/n de contactar a todos", 36, "0,8850", 35, "0.8850", "0,8850", "0.8850", 1e-9, 1, 1),
    ("matriz de confusión con t = 0,15 (fila real 0)", 36, "689", 35, r"re:real 0\s+689\s+111", "689", "689", 1e-9, 1, 1),
    ("matriz de confusión con t = 0,15 (fila real 1)", 36, "55", 35, r"re:real 1\s+49\s+55", "55", "55", 1e-9, 1, 1),
    ("contactados con t = 0,15 en prueba", 36, "166", 35, r"re:166\s+356\s+0\.3938", "166", "166", 1e-9, 1, 1),
    ("sensibilidad con t = 0,15 en prueba (%)", 36, "52,9", 35, "0.5288", "52,9", "0.5288", 5e-2, 1, 100),
    ("especificidad con t = 0,15 en prueba (%)", 36, "86,1", 35, "0.8612", "86,1", "0.8612", 5e-2, 1, 100),
    ("tasa de contratación en el decil más alto (%)", 36, "44", 35, "0.44", "44", "0.44", 1e-9, 1, 100),
    ("proporción positiva de la referencia en prueba", 36, "0,1153", 35, r"re:p=0\.115", "0,1153", "0.115", 5e-4, 1, 1),
    # --- 4(a) PCA (md 40 / código 39) ---
    ("varianza de PC1", 40, "1,1080", 39, "1.1080", "1,1080", "1.1080", 1e-9, 1, 1),
    ("varianza de PC2", 40, "1,0666", 39, "1.0666", "1,0666", "1.0666", 1e-9, 1, 1),
    ("varianza de PC3", 40, "0,9354", 39, "0.9354", "0,9354", "0.9354", 1e-9, 1, 1),
    ("% de varianza de PC1", 40, "27,7", 39, "0.2770", "27,7", "0.2770", 1e-9, 1, 100),
    ("% de varianza de PC2", 40, "26,7", 39, "0.2667", "26,7", "0.2667", 5e-2, 1, 100),
    ("% de varianza de PC3", 40, "23,4", 39, "0.2338", "23,4", "0.2338", 5e-2, 1, 100),
    ("suma PC1 + PC2 (% de la varianza)", 40, "54,4", 39, "0.5437", "54,4", "0.5437", 5e-2, 1, 100),
    ("carga de `edad` en PC1", 40, "0,58", 39, "0.5783", "0,58", "0.5783", 5e-3, 1, 1),
    ("carga de `saldo` en PC1", 40, "0,63", 39, "0.6298", "0,63", "0.6298", 5e-3, 1, 1),
    ("mayor correlación entre las 4 variables", 40, "0,097", 39, "0.097", "0,097", "0.097", 1e-9, 1, 1),
    # --- 4(b) K-means (md 43 / código 42) ---
    ("WSS con k = 2", 43, "11 893", 42, "11892.8666", "11 893", "11892.8666", 1.0, 1, 1),
    ("WSS con k = 3", 43, "9 685", 42, "9684.7306", "9 685", "9684.7306", 1.0, 1, 1),
    ("WSS con k = 4", 43, "7 665", 42, "7665.0127", "7 665", "7665.0127", 1.0, 1, 1),
    ("silueta media con k = 2", 43, "0,301", 42, "0.3014", "0,301", "0.3014", 1e-3, 1, 1),
    ("silueta media con k = 3", 43, "0,333", 42, "0.3335", "0,333", "0.3335", 1e-3, 1, 1),
    ("silueta media con k = 4", 43, "0,363", 42, "0.3631", "0,363", "0.3631", 1e-3, 1, 1),
    ("tamaños de los grupos con k = 4", 43, "2004 / 1289 / 177 / 147", 42,
     r"re:\[2004, 1289, 177, 147\]", None, None, 0, 1, 1),
    ("tasa de contratación del grupo con historial previo (%)", 43, "26,6", 42, "0.266", "26,6", "0.266", 1e-9, 1, 100),
    ("tamaño del grupo 1 (jóvenes)", 43, "2004", 42, "2004", "2004", "2004", 1e-9, 1, 1),
    ("tasa de contratación del grupo 3 (muy insistidos) (%)", 43, "4,8", 42, "0.048", "4,8", "0.048", 1e-9, 1, 100),
    # --- 5(a) Mahalanobis (md 47 / código 46) ---
    ("corte chi-cuadrado (0,99; 4 g.l.)", 47, "13,2767", 46, "13.2767", "13,2767", "13.2767", 1e-9, 1, 1),
    ("registros marcados", 47, "189", 46, "189", "189", "189", 1e-9, 1, 1),
    ("porcentaje de registros marcados", 47, "5,23", 46, "5.23", "5,23", "5.23", 1e-9, 1, 1),
    ("id_cliente del registro más anómalo", 47, "1477", 46, "1477", "1477", "1477", 1e-9, 1, 1),
    ("saldo del registro más anómalo", 47, "71 188", 46, "71188", "71 188", "71188", 1e-9, 1, 1),
    ("d² del registro más anómalo", 47, "518,6", 46, "518.623555", "518,6", "518.623555", 5e-2, 1, 1),
    ("id_cliente del segundo registro", 47, "1542", 46, "1542", "1542", "1542", 1e-9, 1, 1),
    ("contactos_campana del segundo registro", 47, "50", 46, "50", "50", "50", 1e-9, 1, 1),
    ("d² del segundo registro", 47, "224,3", 46, "224.272961", "224,3", "224.272961", 5e-2, 1, 1),
    # --- 1(a)-1(c) preparación (md 12/15/18 / código 11/14/17) ---
    ("duración media de quienes contratan", 12, "554", 11, "554.450839", "554", "554.450839", 1.0, 1, 1),
    ("duración media de quienes no contratan", 12, "228", 11, "227.666250", "228", "227.666250", 1.0, 1, 1),
    ("predictores originales tras excluir", 12, "15", 11, "Predictores originales (15)", "15", "15", 1e-9, 1, 1),
    ("registros de la clase 0", 15, "3200", 14, "3200", "3200", "3200", 1e-9, 1, 1),
    ("registros de la clase 1", 15, "417", 14, "417", "417", "417", 1e-9, 1, 1),
    ("exactitud de la clase mayoritaria (%)", 15, "88,5", 14, "0.8847", "88,5", "0.8847", 5e-2, 1, 100),
    ("unknown en resultado_previo (%)", 15, "82,4", 14, "82.4", "82,4", "82.4", 1e-9, 1, 1),
    ("unknown en canal (%)", 15, "30,1", 14, "30.1", "30,1", "30.1", 1e-9, 1, 1),
    ("registros con dias_previos = -1", 15, "2979", 14, "2979", "2979", "2979", 1e-9, 1, 1),
    ("registros con saldo negativo", 15, "302", 14, "302", "302", "302", 1e-9, 1, 1),
    ("columnas preparadas (p)", 18, "42", 17, "p = 42", "42", "42", 1e-9, 1, 1),
    ("tasa de contratación con contacto previo (%)", 18, "22,4", 17, "0.224", "22,4", "0.224", 1e-9, 1, 100),
    ("tasa de contratación sin contacto previo (%)", 18, "9,2", 17, "0.092", "9,2", "0.092", 1e-9, 1, 100),
    # --- 5(b) contexto de anomalías (md 50 / código 49) ---
    ("asimetría del saldo", 50, "7,06", 49, "7.06", "7,06", "7.06", 1e-9, 1, 1),
    ("asimetría de contactos_campana", 50, "4,91", 49, "4.91", "4,91", "4.91", 1e-9, 1, 1),
    ("z del saldo del registro más anómalo", 50, "22,75", 46, "22.75", "22,75", "22.75", 1e-9, 1, 1),
    ("z de contactos_campana del segundo registro", 50, "14,95", 46, "14.95", "14,95", "14.95", 1e-9, 1, 1),
    ("curtosis en exceso del saldo", 50, "99", 49, "98.82", "99", "98.82", 0.2, 1, 1),
    ("mediana del saldo", 50, "438", 49, "438.0", "438", "438.0", 1e-9, 1, 1),
]

# ---------------------------------------------------------------------------
# Coherencias derivadas: magnitudes del texto recalculadas a partir de los
# propios valores cosechados (descripción, clave, fórmula, tolerancia)
# ---------------------------------------------------------------------------
DERIVADAS = [
    ("sensibilidad OOF en t = 0,15 (%)", "sens_umbral",
     lambda R: 100 * int(R["tp_umbral"]) / int(R["n_positivos"]), 0.6),
    ("especificidad OOF en t = 0,15 (%)", "espec_umbral",
     lambda R: 100 * int(R["tn_umbral"]) / int(R["n_clase0"]), 0.1),
    ("porcentaje de cartera contactada en t = 0,15 (%)", "pct_contactados_umbral",
     lambda R: 100 * int(R["contactados_umbral"]) / int(R["n_registros_entrenamiento"]), 0.6),
    ("tasa de éxito entre los contactados en prueba (%)", "test_tasa_exito",
     lambda R: 100 * int(R["test_tp_015"]) / int(R["test_contactados_015"]), 0.6),
    ("porcentaje de cartera contactada en prueba (%)", "test_pct_cartera",
     lambda R: 100 * int(R["test_contactados_015"]) / int(R["n_test"]), 0.6),
    ("reducción del costo frente a t = 0,50 (%)", "test_reduccion_pct_050",
     lambda R: 100 * (int(R["test_costo_050"]) - int(R["test_costo_015"])) / int(R["test_costo_050"]), 0.6),
    ("mejora del log loss frente a la referencia (%)", "mejora_pct_gb",
     lambda R: 100 * (float(R["ref_log_loss"].replace(",", ".")) - float(R["gb_log_loss"].replace(",", ".")))
     / float(R["ref_log_loss"].replace(",", ".")), 0.6),
    ("C/n en prueba con t = 0,15", "test_cn_015",
     lambda R: int(R["test_costo_015"]) / int(R["n_test"]), 5e-4),
    ("C/n en prueba de la referencia", "test_cn_ref",
     lambda R: int(R["test_costo_ref"]) / int(R["n_test"]), 5e-4),
    ("diferencia de AUC entre las dos logísticas", "delta_auc_logistica",
     lambda R: float(R["log_todas_auc"].replace(",", ".")) - float(R["log10_auc"].replace(",", ".")), 5e-4),
    ("varianza no retenida por PC1+PC2 (%)", "pca_omitido_pct",
     lambda R: 100 - float(R["pca_sum_pct"].replace(",", ".")), 0.1),
]


def buscar(texto: str, evidencia: str) -> bool:
    """Busca la evidencia: literal salvo que empiece por 're:' (expresión regular)."""
    if evidencia.startswith("re:"):
        return re.search(evidencia[3:], texto) is not None
    return evidencia in texto


def main() -> None:
    print("=" * 78)
    print("VERIFICACIÓN DEL NOTEBOOK FINAL")
    print("=" * 78)

    if not NOTEBOOK.exists():
        check(False, "El notebook final existe", f"no se encontró {NOTEBOOK}")
        resumen()
        return
    nb = nbformat.read(NOTEBOOK, as_version=4)
    R = json.loads(RESULTADOS.read_text(encoding="utf-8")) if RESULTADOS.exists() else {}

    mds = [c for c in nb.cells if c.cell_type == "markdown"]
    codes = [c for c in nb.cells if c.cell_type == "code"]

    # --- 1. estructura -----------------------------------------------------
    check(len(nb.cells) == ESPERADOS["total"], "cantidad total de celdas",
          f"{len(nb.cells)} (esperado {ESPERADOS['total']})")
    check(len(mds) == ESPERADOS["markdown"], "cantidad de celdas markdown",
          f"{len(mds)} (esperado {ESPERADOS['markdown']})")
    check(len(codes) == ESPERADOS["code"], "cantidad de celdas de código",
          f"{len(codes)} (esperado {ESPERADOS['code']})")

    con_reporte = [i for i, c in enumerate(nb.cells)
                   if "Celda técnica de recolección" in c.get("source", "")]
    check(not con_reporte, "la celda técnica REPORTE fue eliminada",
          f"celdas con REPORTE: {con_reporte}" if con_reporte else "no está presente")

    marcadores = [i for i, c in enumerate(mds) if "{{" in c.source or "}}" in c.source]
    check(not marcadores, "no quedan marcadores {{...}} en el markdown",
          f"celdas: {marcadores}" if marcadores else "0 marcadores")

    residuos = []
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []):
            t = o.get("text", "")
            t = "".join(t) if isinstance(t, list) else t
            if re.search(r"^@@[^@\n]*@@\s*$", t, re.MULTILINE):
                residuos.append(f"celda {i}")
    check(not residuos, "no quedan residuos @@clave=valor@@ en las salidas",
          ", ".join(residuos) if residuos else "0 residuos")

    errores = [(i, o.get("ename")) for i, c in enumerate(nb.cells)
               for o in c.get("outputs", []) if o.get("output_type") == "error"]
    check(not errores, "ninguna salida es de tipo error", f"{errores}" if errores else "0 errores")

    sin_contador = [i for i, c in enumerate(codes) if c.get("execution_count") is None]
    check(not sin_contador, "toda celda de código tiene execution_count",
          f"celdas sin contador: {sin_contador}" if sin_contador else "todas numeradas")

    sin_salida = [i for i, c in enumerate(codes) if not c.get("outputs")]
    check(not sin_salida, "toda celda de código tiene salida",
          f"celdas sin salida: {sin_salida}" if sin_salida else "todas con salida")

    n_figuras = sum(1 for c in codes for o in c.get("outputs", [])
                    if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}))
    check(n_figuras >= N_FIGURAS_ESPERADAS, "figuras matplotlib presentes",
          f"{n_figuras} figuras (esperadas >= {N_FIGURAS_ESPERADAS})")

    # --- 2. consistencia del render ---------------------------------------
    claves_usadas = set()
    for tipo, src in B.CELLS:
        if tipo == "md":
            claves_usadas.update(re.findall(r"\{\{(\w+)\}\}", src))
    faltantes = sorted(k for k in claves_usadas if k not in R)
    check(not faltantes, "todos los marcadores del texto tienen valor cosechado",
          f"sin valor: {faltantes}" if faltantes else f"{len(claves_usadas)} marcadores con valor")

    if not faltantes:
        esperado_md = [B.render(src, R) for tipo, src in B.CELLS if tipo == "md"]
        difieren = [i for i, (a, b) in enumerate(zip(esperado_md, [c.source for c in mds]))
                    if a != b]
        check(not difieren and len(esperado_md) == len(mds),
              "el markdown entregado es el render exacto de las plantillas + valores",
              f"celdas que difieren: {diferen}" if difieren else
              f"{len(mds)} celdas markdown idénticas al render")

        sin_aparecer = [k for k in sorted(claves_usadas)
                        if str(R[k]) not in "\n".join(m.source for m in mds)]
        check(not sin_aparecer, "cada valor cosechado aparece en el texto",
              f"no aparecen: {sin_aparecer}" if sin_aparecer else
              f"{len(claves_usadas)} valores presentes en la prosa")

    # --- 3. coherencia numérica prosa <-> salidas -------------------------
    for (desc, idx_md, ev_md, idx_cod, ev_out, num_md, num_out, tol, k_md, k_out) in CHECKS:
        if idx_md >= len(nb.cells) or idx_cod >= len(nb.cells):
            check(False, desc, "celda inexistente")
            continue
        md = nb.cells[idx_md].source
        salida = texto_celda(nb.cells[idx_cod])
        ok, detalle = True, ""
        if not buscar(md, ev_md):
            ok, detalle = False, f"'{ev_md}' no aparece en la celda markdown {idx_md}"
        elif not buscar(salida, ev_out):
            ok, detalle = False, f"'{ev_out}' no aparece en la salida de la celda {idx_cod}"
        elif num_md is not None:
            try:
                v_md = a_numero(num_md) * k_md
                v_out = a_numero(num_out) * k_out
            except ValueError as e:
                ok, detalle = False, f"no se pudo interpretar: {e}"
            else:
                if abs(v_md - v_out) > tol:
                    ok = False
                    detalle = f"prosa {v_md:g} != salida {v_out:g} (tol {tol})"
                else:
                    detalle = f"prosa {v_md:g} = salida {v_out:g}"
        else:
            detalle = "presente en ambos lados"
        check(ok, desc, detalle)

    # --- 4. coherencias derivadas -----------------------------------------
    for desc, clave, formula, tol in DERIVADAS:
        if clave not in R:
            check(False, desc, f"falta la clave {clave}")
            continue
        esperado = formula(R)
        obtenido = float(str(R[clave]).replace(",", "."))
        ok = abs(esperado - obtenido) <= tol
        check(ok, desc, f"texto {obtenido:g} vs recálculo {esperado:g} (tol {tol})")

    # --- 5. PDFs de referencia intactos ------------------------------------
    for pdf in PDFS:
        p = CARPETA / pdf
        if not p.exists():
            check(False, f"PDF de referencia presente: {pdf}", "no encontrado")
            continue
        mtime = p.stat().st_mtime
        ref_mtime = (BASE / "build_notebook_v2.py").stat().st_mtime
        check(mtime < ref_mtime, f"PDF de referencia sin modificar: {pdf}",
              f"mtime {mtime:.0f} anterior al de build_notebook_v2.py ({ref_mtime:.0f})")

    resumen()


def resumen() -> None:
    print()
    print("=" * 78)
    print("RESUMEN")
    print("=" * 78)
    ok = sum(1 for r in resultados if r[0])
    for bueno, nombre, detalle in resultados:
        print(f"[{'PASS' if bueno else 'FAIL'}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    print()
    print(f"TOTAL: {ok}/{len(resultados)} comprobaciones PASS")
    if ok != len(resultados):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
