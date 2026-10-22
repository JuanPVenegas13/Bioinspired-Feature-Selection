"""Pruebas genéricas que TODO algoritmo registrado debe pasar.

Usan un problema sintético rápido: 20 variables, de las cuales solo las 3
primeras determinan la clase. Al agregar un algoritmo a ALGORITHMS, estas
pruebas lo cubren automáticamente.
"""

import numpy as np
import pytest

from bfs.algorithms import ALGORITHMS
from bfs.algorithms.cbpso_miws import inertia_weight
from bfs.data import Split
from bfs.fitness import FeatureSelectionProblem

N, T = 10, 30


def make_problem(seed: int = 0) -> FeatureSelectionProblem:
    rng = np.random.default_rng(seed)
    n, dim = 300, 20
    X = rng.random((n, dim))
    y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 1.5).astype(int)
    return FeatureSelectionProblem(Split(X[:240], X[240:], y[:240], y[240:]))


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_result_contract(name):
    problem = make_problem()
    res = ALGORITHMS[name](problem, problem.dim, N, T, np.random.default_rng(1))
    assert res.mask.dtype == bool and res.mask.shape == (problem.dim,)
    assert res.curve.shape == (T,)
    assert res.best_fitness == pytest.approx(problem(res.mask))
    assert res.curve[-1] == pytest.approx(res.best_fitness)
    assert np.all(np.diff(res.curve) <= 1e-12)  # la curva nunca empeora
    assert res.elapsed > 0


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_reproducible_with_same_seed(name):
    a = ALGORITHMS[name](make_problem(), 20, N, T, np.random.default_rng(42))
    b = ALGORITHMS[name](make_problem(), 20, N, T, np.random.default_rng(42))
    np.testing.assert_array_equal(a.mask, b.mask)
    np.testing.assert_array_equal(a.curve, b.curve)


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_finds_informative_features(name):
    problem = make_problem()
    res = ALGORITHMS[name](problem, problem.dim, N, 50, np.random.default_rng(3))
    assert res.mask[:3].sum() >= 2           # encuentra casi todas las relevantes
    assert res.n_selected < problem.dim      # y descarta ruido


def test_inertia_schemes_match_equations():
    rng = np.random.default_rng(0)
    kw = dict(wmax=0.9, wmin=0.4, w0=0.9, p=1.2)
    T = 100
    # IWS 1 y 3 van de ~0.9 a 0.4
    assert inertia_weight(1, 0, T, rng, **kw) == pytest.approx(0.9)
    assert inertia_weight(1, T, T, rng, **kw) == pytest.approx(0.4)
    assert inertia_weight(3, 0, T, rng, **kw) == pytest.approx(0.9)
    assert inertia_weight(3, T, T, rng, **kw) == pytest.approx(0.4)
    # IWS 3 es no lineal: a mitad de camino queda por debajo de la lineal
    assert inertia_weight(3, 50, T, rng, **kw) < inertia_weight(1, 50, T, rng, **kw)
    # IWS 2 en [0.5, 1]; IWS 4 constante
    ws = [inertia_weight(2, 10, T, rng, **kw) for _ in range(1000)]
    assert 0.5 <= min(ws) and max(ws) <= 1.0
    assert inertia_weight(4, 37, T, rng, **kw) == 0.9


def test_v_shaped_transfer():
    from bfs.algorithms.bpso import update_particle, v_shaped

    # V(v) = |tanh(v)|: simétrica, 0 en v = 0, casi 1 en |v| = 6
    assert v_shaped(np.array([0.0]))[0] == 0.0
    assert v_shaped(np.array([-2.0]))[0] == pytest.approx(v_shaped(np.array([2.0]))[0])
    assert v_shaped(np.array([6.0]))[0] > 0.99
    # Si la partícula ya está en pbest = gbest y v = 0, con V NO cambia ningún bit
    # (con la sigmoide cada bit sería un volado)
    x = np.array([1, 0, 1, 1, 0], dtype=np.int8)
    rng = np.random.default_rng(0)
    x_new, v_new = update_particle(x, np.zeros(5), x, x, 0.9, 2, 2, 6, rng, transfer="v")
    np.testing.assert_array_equal(x_new, x)


def test_cbpso_species_and_scheme_selection():
    problem = make_problem()
    res = ALGORITHMS["cbpso_miws"](problem, problem.dim, 10, T, np.random.default_rng(0), ns=3)
    assert res.extra["species_sizes"] == [4, 3, 3]
    hist = res.extra["iws_history"]
    assert hist.shape == (T, 3)
    assert set(np.unique(hist)) <= {1, 2, 3, 4}
    assert len(np.unique(hist)) == 4  # en 90 sorteos aparecen los 4 esquemas
