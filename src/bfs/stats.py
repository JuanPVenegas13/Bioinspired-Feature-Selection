"""Estadística de las mejoras: resumen descriptivo y Wilcoxon pareado.

Requisitos de la materia:
  * reportar **media, mediana y desviación estándar** de las ≥ 30 corridas;
  * comparar con la **prueba de Wilcoxon** (el artículo usó prueba t).

Se usa el Wilcoxon de los rangos con signo **pareado** por corrida: la
corrida ``r`` comparte partición y generador entre métodos (ver
``bfs.experiment``), así que las parejas son legítimas y la prueba es mucho
más potente que su versión no pareada. Al ser no paramétrica no supone
normalidad, que con 30 corridas de fitness acotado no es razonable suponer.

La comparación réplica vs. artículo (Wilcoxon de una muestra) va en
``bfs.replication``, módulo aparte, para no chocar con la etapa 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_METRICS = ("accuracy", "best_fitness", "n_features", "n_evals", "time_s")

LOWER_IS_BETTER = frozenset({
    "best_fitness", "worst_fitness", "mean_fitness", "std_fitness",
    "n_features", "n_evals", "time_s",
})
"""Métricas donde menos es mejor. ``accuracy`` es la excepción: más es mejor."""

ALPHA = 0.05
"""Nivel de significancia."""

REFERENCE = "cbpso_miws"
"""Método de referencia de las comparaciones: la réplica del artículo."""

CHALLENGERS = ("bpso", "cbpso_miws_v", "cbpso_miws_ls", "cbpso_miws_v_ls", "cbpso_miws_T150")
"""Configuraciones que se comparan contra la referencia."""

MAIN_METRICS = ("best_fitness", "accuracy")
"""Métricas principales: lo que el algoritmo optimiza (ec. 11) y la exactitud (ec. 17)."""


def _mejor_es_menor(metric: str) -> bool:
    if metric in LOWER_IS_BETTER:
        return True
    if metric == "accuracy":
        return False
    raise ValueError(
        f"No sé si en {metric!r} es mejor más o menos. Agrégala a LOWER_IS_BETTER "
        "(si menos es mejor) o trátala como accuracy."
    )


# --------------------------------------------------------------------------
# Resumen descriptivo
# --------------------------------------------------------------------------

def summarize(df: pd.DataFrame, metrics=DEFAULT_METRICS) -> pd.DataFrame:
    """Media, mediana y desviación estándar (ddof = 1) por (dataset, método).

    Regresa un DataFrame con índice ``(dataset, method)`` y columnas
    jerárquicas ``(métrica, estadístico)``, más ``("n", "")`` con el número
    de corridas (debe ser 30).
    """
    metrics = list(metrics)
    faltan = [m for m in metrics if m not in df.columns]
    if faltan:
        raise KeyError(f"El DataFrame no tiene las columnas {faltan}")

    g = df.groupby(["dataset", "method"], sort=True)
    out = g[metrics].agg(["mean", "median", "std"])     # std de pandas: ddof = 1
    out[("n", "")] = g["run"].nunique()
    return out


def summary_table(df: pd.DataFrame, metric: str = "accuracy", decimals: int = 4) -> pd.DataFrame:
    """Tabla ``media ± desv.`` (y mediana) de UNA métrica, bases en filas.

    Pensada para el póster: un cuadro por métrica, legible de un vistazo.
    """
    g = df.groupby(["dataset", "method"])[metric]
    texto = (
        g.mean().round(decimals).astype(str)
        + " ± " + g.std().round(decimals).astype(str)
        + " (" + g.median().round(decimals).astype(str) + ")"
    )
    return texto.unstack("method")


# --------------------------------------------------------------------------
# Wilcoxon pareado
# --------------------------------------------------------------------------

TOL = 1e-12
"""Diferencias con |d| <= TOL se consideran empates exactos.

``runs.csv`` guarda los flotantes con ~16 cifras significativas, así que un
mismo valor puede reaparecer con un error de ~1e-17 (por ejemplo, al comparar
filas leídas del CSV contra filas recién corridas). Sin este umbral, esas
diferencias numéricas —que representan **la misma solución**— entrarían al
ranking de Wilcoxon como si fueran diferencias reales, ocupando los rangos más
chicos y corriendo el estadístico. Con el umbral se descartan como empates,
que es lo que son.
"""


def wilcoxon_paired(
    df: pd.DataFrame, metric: str, method_a: str, method_b: str,
    alpha: float = ALPHA, tol: float = TOL,
) -> pd.DataFrame:
    """Wilcoxon de rangos con signo, pareado por ``run``, una fila por base.

    Columnas: ``dataset``, ``n``, ``median_diff``, ``statistic``, ``p_value``,
    ``ganador`` (``'a'``, ``'b'`` o ``'empate'`` si ``p >= alpha``).

    ``median_diff`` es la mediana de las diferencias ``a - b`` por corrida.
    El ganador se decide por el signo de esa diferencia y por la dirección de
    la métrica (en ``accuracy`` más es mejor; en el resto, menos).

    Si **todas** las diferencias son cero (pasa en las bases de 9 variables,
    donde todas las variantes llegan al mismo óptimo) ``scipy`` no puede
    calcular la prueba: se regresa ``p_value = 1.0`` y ``ganador = "empate"``.

    Las diferencias con ``|d| <= tol`` se tratan como cero exacto (ver ``TOL``).
    """
    for m in (method_a, method_b):
        if m not in set(df["method"]):
            raise KeyError(f"El método {m!r} no está en el DataFrame")
    if metric not in df.columns:
        raise KeyError(f"El DataFrame no tiene la columna {metric!r}")
    menor_es_mejor = _mejor_es_menor(metric)

    filas = []
    for dataset, sub in df.groupby("dataset", sort=True):
        # Pareo por corrida: solo las corridas presentes en ambos métodos.
        ancho = sub.pivot_table(index="run", columns="method", values=metric)
        if method_a not in ancho or method_b not in ancho:
            continue
        par = ancho[[method_a, method_b]].dropna()
        a, b = par[method_a].to_numpy(float), par[method_b].to_numpy(float)
        dif = a - b
        dif[np.abs(dif) <= tol] = 0.0     # ruido de punto flotante = empate
        n = len(dif)

        # zero_method="wilcox": los empates se descartan y n se reduce. Se hace
        # aquí en vez de dejárselo a scipy porque scipy, al ver ceros, se pasa a
        # la aproximación normal; sin ceros usa la prueba EXACTA, que es la
        # correcta con pocas corridas (y aquí sobran empates en las bases de 9
        # variables). El estadístico es el mismo; el p-valor, más preciso.
        sin_empates = dif[dif != 0.0]

        if n == 0 or sin_empates.size == 0:
            # Sin diferencias que ordenar: la prueba no aplica.
            estadistico, p = 0.0, 1.0
        else:
            res = stats.wilcoxon(sin_empates)
            estadistico, p = float(res.statistic), float(res.pvalue)

        mediana = float(np.median(dif))
        ventaja = mediana if mediana != 0.0 else float(np.mean(dif))
        if p >= alpha or ventaja == 0.0:
            ganador = "empate"
        elif (ventaja < 0.0) == menor_es_mejor:
            ganador = "a"
        else:
            ganador = "b"

        filas.append({
            "dataset": dataset, "n": n, "median_diff": mediana,
            "statistic": estadistico, "p_value": p, "ganador": ganador,
        })

    return pd.DataFrame(filas, columns=["dataset", "n", "median_diff", "statistic", "p_value", "ganador"])


# --------------------------------------------------------------------------
# Comparaciones múltiples
# --------------------------------------------------------------------------

def holm(p_values) -> np.ndarray:
    """Corrección de Holm-Bonferroni (control del error familiar, FWER).

    Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    Scandinavian Journal of Statistics, 6(2), 65-70.

    Se ordenan los m p-valores de menor a mayor y el i-ésimo se multiplica por
    ``m - i``; luego se impone monotonía acumulada y se acota a 1. Es uniformemente
    más potente que Bonferroni y, a diferencia de Benjamini-Hochberg
    (``scipy.stats.false_discovery_control``), controla el FWER, no el FDR.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p.copy()
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("Los p-valores deben estar en [0, 1] y no ser NaN")

    m = p.size
    orden = np.argsort(p, kind="stable")
    ajustados = np.maximum.accumulate((m - np.arange(m)) * p[orden])
    salida = np.empty(m)
    salida[orden] = np.minimum(ajustados, 1.0)
    return salida


def compare_all(
    df: pd.DataFrame,
    reference: str = REFERENCE,
    challengers=CHALLENGERS,
    metrics=MAIN_METRICS,
    alpha: float = ALPHA,
    correction: str | None = "holm",
) -> pd.DataFrame:
    """Todas las comparaciones principales contra ``reference``, en una tabla.

    Una fila por (métrica, base, configuración). ``resultado`` está escrito
    desde el punto de vista del **retador**: ``"mejor"`` / ``"peor"`` /
    ``"empate"`` frente a la referencia.

    Con ``correction="holm"`` se agregan ``p_holm`` y ``resultado_holm``,
    corrigiendo por el número de comparaciones **dentro de cada métrica**
    (todas las bases × todos los retadores). ``correction=None`` lo omite.
    """
    if correction not in (None, "holm"):
        raise ValueError(f"correction debe ser 'holm' o None, recibí {correction!r}")

    presentes = set(df["method"])
    challengers = [m for m in challengers if m in presentes]

    partes = []
    for metric in metrics:
        bloque = []
        for retador in challengers:
            tabla = wilcoxon_paired(df, metric, reference, retador, alpha=alpha)
            tabla.insert(0, "metric", metric)
            tabla.insert(1, "reference", reference)
            tabla.insert(2, "method", retador)
            # ganador 'a' = gana la referencia -> el retador es peor
            tabla["resultado"] = tabla["ganador"].map(
                {"a": "peor", "b": "mejor", "empate": "empate"}
            )
            bloque.append(tabla)
        if not bloque:
            continue
        bloque = pd.concat(bloque, ignore_index=True)

        if correction == "holm":
            bloque["p_holm"] = holm(bloque["p_value"])
            bloque["resultado_holm"] = np.where(
                bloque["p_holm"] >= alpha, "empate", bloque["resultado"]
            )
        partes.append(bloque)

    if not partes:
        return pd.DataFrame(columns=["metric", "reference", "method", "dataset", "n",
                                     "median_diff", "statistic", "p_value", "ganador",
                                     "resultado"])
    out = pd.concat(partes, ignore_index=True)
    return out.drop(columns="ganador")


def pvalue_matrix(comparaciones: pd.DataFrame, metric: str, columna: str = "p_value") -> pd.DataFrame:
    """Matriz bases × configuraciones de p-valores, para la tabla del póster."""
    sub = comparaciones[comparaciones["metric"] == metric]
    return sub.pivot(index="dataset", columns="method", values=columna)
