"""Óptimo exacto por enumeración (etapa 3).

En las bases con pocas variables se puede evaluar TODO el espacio de búsqueda:
con D variables hay 2^D - 1 subconjuntos no vacíos (511 con D = 9, 1,023 con
D = 10). Para cada partición 80/20 (``seed = run``) se evalúan todos con la
misma función de fitness que usan los algoritmos (ecs. 11-12), lo que da el
**óptimo global real** de esa corrida.

Con eso se mide algo que el artículo no puede medir: en cuántas de las 30
corridas cada algoritmo encontró la mejor solución posible ("tasa de acierto")
y a qué distancia quedó cuando no la encontró ("brecha").

Con D = 34 (Ionosphere) habría ~1.7e10 subconjuntos: imposible enumerarlos,
y justamente por eso se usan metaheurísticas.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from bfs.data import holdout_split, load_dataset
from bfs.fitness import FeatureSelectionProblem

MAX_DIM = 12
"""Límite de seguridad: 2^12 = 4,096 subconjuntos por corrida."""

TOL = 1e-12
"""Tolerancia numérica para decidir si un fitness iguala al óptimo."""

SMALL_DATASETS = ("breast_cancer_wisconsin", "glass", "glass_id", "breast_cancer_coimbra")
"""Bases enumerables (9 o 10 variables)."""


def all_masks(dim: int) -> np.ndarray:
    """Las 2^dim - 1 máscaras no vacías, una por fila (int8)."""
    if dim > MAX_DIM:
        raise ValueError(f"dim = {dim}: 2^{dim} subconjuntos es demasiado para enumerar")
    masks = np.array(list(itertools.product((0, 1), repeat=dim)), dtype=np.int8)
    return masks[masks.sum(axis=1) > 0]


def exact_optimum(problem: FeatureSelectionProblem) -> dict:
    """Evalúa todos los subconjuntos de ``problem`` y regresa el óptimo.

    Puede haber empates (varios subconjuntos con el mismo fitness mínimo); se
    reportan todos en ``optimal_masks``.
    """
    masks = all_masks(problem.dim)
    fits = np.array([problem(m) for m in masks])
    best = fits.min()
    optimos = masks[fits <= best + TOL]
    primero = optimos[0]
    return {
        "opt_fitness": float(best),
        "n_optimal": int(len(optimos)),
        "opt_accuracy": float(problem.accuracy(primero)),
        "opt_n_features": int(primero.sum()),
        "opt_selected": ";".join(str(i) for i in np.flatnonzero(primero)),
        "n_subsets": int(len(masks)),
        "optimal_masks": optimos,
    }


def compute_optima(datasets=SMALL_DATASETS, runs: int = 30, verbose: bool = True) -> pd.DataFrame:
    """Óptimo exacto de cada (base, corrida). Usa ``holdout_split(seed=run)``,
    la misma partición que el runner, así que es comparable fila a fila con ``runs.csv``."""
    filas = []
    for dataset in datasets:
        ds = load_dataset(dataset)
        for r in range(runs):
            out = exact_optimum(FeatureSelectionProblem(holdout_split(ds, seed=r)))
            out.pop("optimal_masks")
            filas.append({"dataset": dataset, "run": r, **out})
        if verbose:
            print(f"[óptimo exacto] {dataset}: {runs} corridas × {filas[-1]['n_subsets']} subconjuntos")
    return pd.DataFrame(filas)


def rescore(runs: pd.DataFrame) -> pd.DataFrame:
    """Re-evalúa en ESTA máquina la máscara final de cada corrida (columna ``selected``).

    Por qué: KNN desempata vecinos a la misma distancia de forma que puede
    variar entre procesadores (pasa en bases con variables enteras, como
    Wisconsin y Coimbra). Así, el ``best_fitness`` guardado en ``runs.csv``
    (calculado en otra computadora) puede diferir ligeramente del que da esta
    máquina para la misma máscara. Para comparar contra el óptimo exacto con la
    MISMA función de fitness, se re-evalúa cada máscara aquí.

    Agrega las columnas ``fitness_here`` y ``differs`` (True si no coincide
    con ``best_fitness``).
    """
    problemas: dict[tuple[str, int], FeatureSelectionProblem] = {}
    datasets = {d: load_dataset(d) for d in runs["dataset"].unique()}
    fits = []
    for d, r, sel in runs[["dataset", "run", "selected"]].itertuples(index=False):
        if (d, r) not in problemas:
            problemas[(d, r)] = FeatureSelectionProblem(holdout_split(datasets[d], seed=r))
        p = problemas[(d, r)]
        mask = np.zeros(p.dim, dtype=bool)
        if isinstance(sel, str) and sel:
            mask[[int(i) for i in sel.split(";")]] = True
        fits.append(p(mask))
    out = runs.copy()
    out["fitness_here"] = fits
    out["differs"] = ~np.isclose(out["fitness_here"], out["best_fitness"], atol=1e-9)
    return out


def hit_rate(runs: pd.DataFrame, optima: pd.DataFrame) -> pd.DataFrame:
    """Por (base, método): % de corridas que alcanzan el óptimo y brecha promedio.

    Usa el fitness re-evaluado en esta máquina (ver ``rescore``).
    Columnas: dataset, method, n, hits, hit_rate (%), gap_mean, gap_max,
    n_differs (corridas cuyo fitness guardado no se reprodujo aquí).
    """
    pequenas = runs[runs["dataset"].isin(optima["dataset"].unique())]
    m = rescore(pequenas).merge(optima[["dataset", "run", "opt_fitness"]], on=["dataset", "run"])
    gap = m["fitness_here"] - m["opt_fitness"]
    if (gap < -1e-9).any():
        raise AssertionError("Una máscara superó al óptimo exacto: revisa particiones/semillas")
    m = m.assign(gap=gap.clip(lower=0), hit=gap <= TOL)
    out = (m.groupby(["dataset", "method"])
             .agg(n=("hit", "size"), hits=("hit", "sum"),
                  gap_mean=("gap", "mean"), gap_max=("gap", "max"),
                  n_differs=("differs", "sum"))
             .reset_index())
    out["hit_rate"] = 100 * out["hits"] / out["n"]
    return out[["dataset", "method", "n", "hits", "hit_rate", "gap_mean", "gap_max", "n_differs"]]
