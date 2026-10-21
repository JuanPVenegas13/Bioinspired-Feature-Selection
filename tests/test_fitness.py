"""Pruebas de la función de fitness con datos sintéticos (no requieren UCI)."""

import numpy as np
import pytest

from bfs.data import Split
from bfs.fitness import EMPTY_SUBSET_FITNESS, FeatureSelectionProblem


@pytest.fixture
def problem():
    # La variable 0 separa perfectamente las clases; las otras 3 son ruido.
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    X = rng.random((n, 4))
    X[:, 0] = y + 0.05 * rng.random(n)
    split = Split(X[:160], X[160:], y[:160], y[160:])
    return FeatureSelectionProblem(split, k=5, alpha=0.01)


def test_perfect_feature_gives_only_size_penalty(problem):
    mask = np.array([1, 0, 0, 0])
    assert problem.error(mask) == 0.0
    assert problem(mask) == pytest.approx(0.01 * 1 / 4)  # alpha * |S|/|T|


def test_noise_features_are_worse(problem):
    assert problem(np.array([0, 1, 1, 1])) > problem(np.array([1, 0, 0, 0]))


def test_formula_matches_equation_11(problem):
    mask = np.array([1, 1, 0, 1])
    err = problem.error(mask)
    assert problem(mask) == pytest.approx(0.01 * 3 / 4 + 0.99 * err)


def test_empty_mask(problem):
    assert problem(np.zeros(4)) == EMPTY_SUBSET_FITNESS


def test_accuracy_is_percentage(problem):
    assert problem.accuracy(np.array([1, 0, 0, 0])) == 100.0


def test_cache_and_eval_counter(problem):
    mask = np.array([1, 0, 1, 0])
    a = problem(mask)
    b = problem(mask.astype(bool))  # mismo subconjunto, otro dtype
    assert a == b
    assert problem.n_evals == 2      # cuenta llamadas aunque use caché
    assert len(problem._cache) == 1  # pero el KNN solo se entrenó una vez
