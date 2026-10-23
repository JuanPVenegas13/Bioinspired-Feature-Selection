"""Pruebas del óptimo exacto por enumeración."""

import numpy as np
import pandas as pd
import pytest

from bfs.data import Split
from bfs.exact import all_masks, compute_optima, exact_optimum, hit_rate
from bfs.fitness import FeatureSelectionProblem


def test_all_masks():
    m = all_masks(4)
    assert m.shape == (15, 4)                      # 2^4 - 1, sin la vacía
    assert len({tuple(x) for x in m}) == 15        # todas distintas
    with pytest.raises(ValueError):
        all_masks(30)


def test_exact_optimum_finds_the_informative_feature():
    # La variable 0 separa las clases; las otras 4 son ruido -> óptimo = {0}
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    X = rng.random((n, 5))
    X[:, 0] = y + 0.05 * rng.random(n)
    problem = FeatureSelectionProblem(Split(X[:160], X[160:], y[:160], y[160:]))
    out = exact_optimum(problem)
    assert out["opt_selected"] == "0"
    assert out["opt_fitness"] == pytest.approx(0.01 * 1 / 5)
    assert out["n_subsets"] == 31


@pytest.fixture(scope="module")
def optima():
    return compute_optima(("breast_cancer_coimbra",), runs=2, verbose=False)


def test_hit_rate(optima):
    # Corrida 0 con la máscara óptima (acierta); corrida 1 con una sola variable (no acierta, salvo que sea el óptimo)
    runs = pd.DataFrame([
        {"dataset": "breast_cancer_coimbra", "method": "x", "run": 0,
         "best_fitness": optima.loc[0, "opt_fitness"], "selected": optima.loc[0, "opt_selected"]},
        {"dataset": "breast_cancer_coimbra", "method": "x", "run": 1,
         "best_fitness": np.nan, "selected": "8"},
    ])
    t = hit_rate(runs, optima).iloc[0]
    acierto_1 = optima.loc[1, "opt_selected"] == "8"
    assert t["hits"] == 1 + acierto_1
    assert t["gap_max"] >= 0
