"""Pruebas de las configuraciones de la etapa 3 (sensibilidad a ns y ablación)."""

import pytest

from bfs.experiment import ALL_METHODS, ETAPA3_METHODS, METHODS, run_one


def test_etapa3_methods_are_separate():
    # No se mezclan con la corrida por defecto ni con el notebook 03
    assert not set(ETAPA3_METHODS) & set(METHODS)
    assert set(ALL_METHODS) == set(METHODS) | set(ETAPA3_METHODS)


@pytest.mark.parametrize("method", sorted(ETAPA3_METHODS))
def test_etapa3_method_runs(method):
    row, curve, evals = run_one("lung_cancer", method, run=0, T=3)
    assert row["method"] == method
    assert len(curve) == 3
    assert row["best_fitness"] == curve.min()
    assert row["n_evals"] == 10 + 10 * 3  # N inicial + N por iteración


def test_ns_controls_species_sizes():
    from bfs.algorithms import ALGORITHMS
    from bfs.data import holdout_split, load_dataset
    from bfs.fitness import FeatureSelectionProblem
    import numpy as np

    ds = load_dataset("lung_cancer")
    for ns, sizes in [(1, [10]), (4, [3, 3, 2, 2]), (5, [2, 2, 2, 2, 2])]:
        p = FeatureSelectionProblem(holdout_split(ds, seed=0))
        res = ALGORITHMS["cbpso_miws"](p, p.dim, 10, 2, np.random.default_rng(0), ns=ns)
        assert res.extra["species_sizes"] == sizes


def test_single_scheme_ablation():
    from bfs.algorithms import ALGORITHMS
    from bfs.data import holdout_split, load_dataset
    from bfs.fitness import FeatureSelectionProblem
    import numpy as np

    ds = load_dataset("lung_cancer")
    p = FeatureSelectionProblem(holdout_split(ds, seed=0))
    res = ALGORITHMS["cbpso_miws"](p, p.dim, 10, 5, np.random.default_rng(0), schemes=(3,))
    assert set(res.extra["iws_history"].ravel()) == {3}
