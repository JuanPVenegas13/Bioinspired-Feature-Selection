"""Réplica vs. artículo (etapa 3).

El artículo (Too et al., 2019, Tablas 3 y 5) solo publica el PROMEDIO de sus
20 corridas, no las corridas individuales. Por eso la comparación estadística
réplica vs. artículo es una prueba de Wilcoxon de rangos con signo de UNA
muestra: nuestras 30 corridas contra el valor reportado,

    H0: la mediana de (nuestro valor - valor del artículo) es 0.

Los valores del artículo están en ``data/paper_results.csv`` (transcritos de
las Tablas 3 y 5 para nuestras 6 bases). Glass del artículo se compara contra
nuestras dos versiones: ``glass`` (9 variables) y ``glass_id`` (10, con ID).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from bfs.stats import holm

PAPER_CSV = Path(__file__).resolve().parents[2] / "data" / "paper_results.csv"

PAPER_METHOD = {"bpso": "BPSO", "cbpso_miws": "CBPSO-MIWS"}
"""Nuestro nombre de método -> nombre en el artículo."""

ALIASES = {"glass_id": "glass"}
"""Bases nuestras que se comparan contra otra fila del artículo."""

METRICS = ("accuracy", "best_fitness", "mean_fitness", "n_features")


def load_paper_results(path: Path | str | None = None) -> pd.DataFrame:
    """Tablas 3 y 5 del artículo, una fila por (base, método)."""
    return pd.read_csv(path or PAPER_CSV)


def paper_value(paper: pd.DataFrame, dataset: str, method: str, metric: str) -> float:
    """Valor reportado por el artículo para nuestra base y método."""
    fila = paper[(paper["dataset"] == ALIASES.get(dataset, dataset))
                 & (paper["method"] == PAPER_METHOD[method])]
    if fila.empty:
        raise KeyError(f"El artículo no reporta {dataset!r} / {method!r}")
    return float(fila[metric].iloc[0])


def wilcoxon_one_sample(x: np.ndarray, value: float) -> tuple[float, float]:
    """Wilcoxon de una muestra de ``x`` contra ``value``. Regresa (estadístico, p).

    Si todas las diferencias son 0 (todas las corridas igualan al valor), la
    prueba no está definida: se regresa p = 1 (no hay evidencia de diferencia).
    """
    d = np.asarray(x, dtype=float) - value
    if np.allclose(d, 0):
        return 0.0, 1.0
    res = wilcoxon(d, zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)


def compare_with_paper(
    runs: pd.DataFrame,
    methods: tuple[str, ...] = ("cbpso_miws", "bpso"),
    metrics: tuple[str, ...] = METRICS,
    paper: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Una fila por (método, métrica, base): nuestra media/mediana/desv. vs. el artículo.

    Columnas: method, metric, dataset, n, ours_mean, ours_median, ours_std,
    paper, diff (ours_mean - paper), statistic, p_value, p_holm, resultado.
    ``resultado`` (con Holm dentro de cada método y métrica): 'igual' si no hay
    diferencia significativa; 'mayor' / 'menor' si la mediana de la réplica
    queda por encima / por debajo del artículo.
    """
    paper = load_paper_results() if paper is None else paper
    filas = []
    for method in methods:
        for metric in metrics:
            bloque = []
            for dataset, g in runs[runs["method"] == method].groupby("dataset"):
                x = g.sort_values("run")[metric].to_numpy(dtype=float)
                ref = paper_value(paper, dataset, method, metric)
                stat, p = wilcoxon_one_sample(x, ref)
                bloque.append({
                    "method": method, "metric": metric, "dataset": dataset,
                    "n": len(x), "ours_mean": x.mean(), "ours_median": np.median(x),
                    "ours_std": x.std(ddof=1), "paper": ref,
                    "diff": x.mean() - ref, "statistic": stat, "p_value": p,
                    "_mediana_d": np.median(x - ref),
                })
            p_holm = holm([b["p_value"] for b in bloque])
            for b, ph in zip(bloque, p_holm):
                b["p_holm"] = float(ph)
                if ph >= alpha:
                    b["resultado"] = "igual"
                else:
                    b["resultado"] = "mayor" if b["_mediana_d"] > 0 else "menor"
                del b["_mediana_d"]
                filas.append(b)
    return pd.DataFrame(filas)


def side_by_side(runs: pd.DataFrame, metric: str = "accuracy",
                 paper: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla para el póster: por base, artículo vs. réplica (media ± desv.) para BPSO y CBPSO-MIWS."""
    paper = load_paper_results() if paper is None else paper
    filas = []
    for dataset in sorted(runs["dataset"].unique()):
        fila = {"dataset": dataset}
        for method, nombre in PAPER_METHOD.items():
            x = runs[(runs["dataset"] == dataset) & (runs["method"] == method)][metric]
            fila[f"{nombre} artículo"] = paper_value(paper, dataset, method, metric)
            fila[f"{nombre} réplica"] = f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"
        filas.append(fila)
    return pd.DataFrame(filas).set_index("dataset")
