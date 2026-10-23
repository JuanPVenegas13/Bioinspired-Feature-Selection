"""Pruebas de la mejora 2: búsqueda local sobre el Gbest (bit-flip hill climbing).

Las pruebas genéricas del contrato (``tests/test_algorithms.py``) ya cubren
``cbpso_miws_ls`` y ``cbpso_miws_v_ls`` por estar en ALGORITHMS. Aquí se prueba
la búsqueda local en sí y, sobre todo, que la **réplica no cambió**.
"""

import numpy as np
import pytest

from bfs.algorithms import ALGORITHMS
from bfs.algorithms.cbpso_miws import run as run_cbpso_miws
from bfs.algorithms.local_search import bit_flip_local_search
from bfs.data import Split
from bfs.fitness import FeatureSelectionProblem

N, T = 10, 30


def make_problem(seed: int = 0) -> FeatureSelectionProblem:
    """Mismo problema sintético que test_algorithms: 20 variables, 3 informativas."""
    rng = np.random.default_rng(seed)
    n, dim = 300, 20
    X = rng.random((n, dim))
    y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 1.5).astype(int)
    return FeatureSelectionProblem(Split(X[:240], X[240:], y[:240], y[240:]))


class ContadorFitness:
    """Envuelve una función de fitness para contar las evaluaciones."""

    def __init__(self, fn):
        self.fn = fn
        self.n = 0

    def __call__(self, mask):
        self.n += 1
        return self.fn(mask)


# -- bit_flip_local_search ----------------------------------------------------

def test_nunca_empeora():
    """La búsqueda local es monótona: jamás regresa algo peor que la entrada."""
    problem = make_problem()
    rng = np.random.default_rng(0)
    for _ in range(30):
        mask = (rng.random(problem.dim) > 0.5).astype(np.int8)
        fit = problem(mask)
        nueva, nuevo_fit = bit_flip_local_search(mask, fit, problem, rng, n_flips=5)
        assert nuevo_fit <= fit
        assert nuevo_fit == pytest.approx(problem(nueva))


def test_gasta_a_lo_mas_n_flips_evaluaciones():
    problem = make_problem()
    for n_flips in (1, 3, 5, 20):
        contador = ContadorFitness(problem)
        mask = np.ones(problem.dim, dtype=np.int8)
        bit_flip_local_search(mask, problem(mask), contador, np.random.default_rng(7), n_flips)
        assert contador.n <= n_flips


def test_n_flips_se_acota_a_dim():
    """n_flips > dim no explota ni repite bits."""
    problem = make_problem()
    contador = ContadorFitness(problem)
    mask = np.ones(problem.dim, dtype=np.int8)
    bit_flip_local_search(mask, problem(mask), contador, np.random.default_rng(0), n_flips=999)
    assert contador.n <= problem.dim


def test_reproducible_con_la_misma_semilla():
    problem = make_problem()
    mask = np.ones(problem.dim, dtype=np.int8)
    fit = problem(mask)
    a = bit_flip_local_search(mask, fit, problem, np.random.default_rng(42), n_flips=5)
    b = bit_flip_local_search(mask, fit, problem, np.random.default_rng(42), n_flips=5)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1]


def test_no_modifica_la_mascara_de_entrada():
    problem = make_problem()
    mask = np.ones(problem.dim, dtype=np.int8)
    original = mask.copy()
    bit_flip_local_search(mask, problem(mask), problem, np.random.default_rng(1), n_flips=20)
    np.testing.assert_array_equal(mask, original)


def test_encuentra_el_flip_que_mejora():
    """Problema sintético donde apagar un bit concreto mejora el fitness.

    El fitness es mínimo en la máscara objetivo; partimos de ella con el bit 3
    encendido de más, así que el único volteo que mejora es apagar el bit 3.
    """
    objetivo = np.array([1, 1, 0, 0, 0], dtype=np.int8)

    def fitness(mask):
        return float(np.sum(np.asarray(mask, dtype=np.int8) != objetivo))

    mask = np.array([1, 1, 0, 1, 0], dtype=np.int8)   # sobra el bit 3
    fit = fitness(mask)                                # = 1.0
    nueva, nuevo_fit = bit_flip_local_search(mask, fit, fitness, np.random.default_rng(0), n_flips=5)
    np.testing.assert_array_equal(nueva, objetivo)
    assert nuevo_fit == 0.0


def test_primera_mejora_regresa_de_inmediato():
    """Con criterio de primera mejora se acepta el primer vecino mejor, no el óptimo."""
    orden = np.random.default_rng(0).permutation(5)

    def fitness(mask):
        # Cualquier bit apagado mejora; el mejor sería apagar todos.
        return float(np.asarray(mask, dtype=np.int8).sum())

    mask = np.ones(5, dtype=np.int8)
    nueva, nuevo_fit = bit_flip_local_search(mask, 5.0, fitness, np.random.default_rng(0), n_flips=5)
    assert nuevo_fit == 4.0                      # un solo bit, no cinco
    assert nueva[orden[0]] == 0                  # justamente el primero de la permutación


def test_si_nada_mejora_regresa_lo_mismo():
    def fitness(mask):
        return float(np.sum(np.asarray(mask, dtype=np.int8) != np.zeros(5, dtype=np.int8)))

    mask = np.zeros(5, dtype=np.int8)            # ya es el óptimo global
    nueva, nuevo_fit = bit_flip_local_search(mask, 0.0, fitness, np.random.default_rng(0), n_flips=5)
    np.testing.assert_array_equal(nueva, mask)
    assert nuevo_fit == 0.0


# -- la réplica no cambió -----------------------------------------------------

def test_gbest_hook_none_es_identico_a_la_replica():
    """Con gbest_hook=None el Result y el consumo de rng son EXACTAMENTE los de antes."""
    p_sin = make_problem()
    sin = run_cbpso_miws(p_sin, p_sin.dim, N, T, np.random.default_rng(5))

    p_con = make_problem()
    con = run_cbpso_miws(p_con, p_con.dim, N, T, np.random.default_rng(5), gbest_hook=None)

    np.testing.assert_array_equal(sin.mask, con.mask)
    np.testing.assert_array_equal(sin.curve, con.curve)
    assert sin.best_fitness == con.best_fitness
    assert p_sin.n_evals == p_con.n_evals
    np.testing.assert_array_equal(sin.extra["iws_history"], con.extra["iws_history"])


def test_el_rng_queda_en_el_mismo_estado_sin_hook():
    """Prueba fuerte de 'no cambió': el generador se consume igual."""
    rng_sin, rng_con = np.random.default_rng(11), np.random.default_rng(11)
    p = make_problem()
    run_cbpso_miws(p, p.dim, N, T, rng_sin)
    p = make_problem()
    run_cbpso_miws(p, p.dim, N, T, rng_con, gbest_hook=None)
    assert rng_sin.random() == rng_con.random()


# -- configuraciones registradas ---------------------------------------------

@pytest.mark.parametrize("name", ["cbpso_miws_ls", "cbpso_miws_v_ls"])
def test_registrados_en_algorithms(name):
    assert name in ALGORITHMS


@pytest.mark.parametrize("name", ["cbpso_miws_ls", "cbpso_miws_v_ls"])
def test_la_busqueda_local_gasta_evaluaciones_extra(name):
    """Cota del costo: a lo más n_flips evaluaciones extra por iteración."""
    p_base = make_problem()
    ALGORITHMS["cbpso_miws"](p_base, p_base.dim, N, T, np.random.default_rng(2))

    p_ls = make_problem()
    ALGORITHMS[name](p_ls, p_ls.dim, N, T, np.random.default_rng(2), n_flips=5)

    assert p_ls.n_evals > p_base.n_evals
    assert p_ls.n_evals <= p_base.n_evals + 5 * T


@pytest.mark.parametrize("name", ["cbpso_miws_ls", "cbpso_miws_v_ls"])
def test_n_flips_es_configurable(name):
    p = make_problem()
    ALGORITHMS[name](p, p.dim, N, T, np.random.default_rng(2), n_flips=1)
    base = make_problem()
    ALGORITHMS["cbpso_miws"](base, base.dim, N, T, np.random.default_rng(2))
    assert p.n_evals <= base.n_evals + T


def test_ls_conserva_los_parametros_del_articulo():
    """La mejora 2 no toca especies ni inercia múltiple (Tabla 2)."""
    p = make_problem()
    res = ALGORITHMS["cbpso_miws_ls"](p, p.dim, 10, T, np.random.default_rng(0))
    assert res.extra["species_sizes"] == [4, 3, 3]
    assert res.extra["iws_history"].shape == (T, 3)


# -- BitFlipHook: contabilidad del costo -------------------------------------

def test_hook_registra_una_entrada_por_iteracion():
    """cbpso_miws llama al hook exactamente una vez por iteración."""
    from bfs.algorithms.local_search import BitFlipHook

    problem = make_problem()
    hook = BitFlipHook(n_flips=5)
    run_cbpso_miws(problem, problem.dim, N, T, np.random.default_rng(0), gbest_hook=hook)
    assert len(hook.evals) == T
    assert all(0 <= e <= 5 for e in hook.evals)


def test_hook_cuadra_con_n_evals():
    """N iniciales + N por iteración + lo que gastó la búsqueda local."""
    from bfs.algorithms.local_search import BitFlipHook

    problem = make_problem()
    hook = BitFlipHook(n_flips=5)
    run_cbpso_miws(problem, problem.dim, N, T, np.random.default_rng(0), gbest_hook=hook)
    assert problem.n_evals == N + N * T + sum(hook.evals)


def test_hook_es_equivalente_a_partial():
    """Observar no cambia nada: mismo Result que con functools.partial."""
    from functools import partial

    from bfs.algorithms.local_search import BitFlipHook

    p_hook = make_problem()
    con_hook = run_cbpso_miws(p_hook, p_hook.dim, N, T, np.random.default_rng(9),
                              gbest_hook=BitFlipHook(n_flips=5))
    p_partial = make_problem()
    con_partial = run_cbpso_miws(
        p_partial, p_partial.dim, N, T, np.random.default_rng(9),
        gbest_hook=partial(bit_flip_local_search, n_flips=5),
    )
    np.testing.assert_array_equal(con_hook.mask, con_partial.mask)
    np.testing.assert_array_equal(con_hook.curve, con_partial.curve)
    assert p_hook.n_evals == p_partial.n_evals


@pytest.mark.parametrize("name", ["cbpso_miws_ls", "cbpso_miws_v_ls"])
def test_extra_trae_ls_evals(name):
    problem = make_problem()
    res = ALGORITHMS[name](problem, problem.dim, N, T, np.random.default_rng(0), n_flips=4)
    assert res.extra["ls_evals"].shape == (T,)
    assert res.extra["n_flips"] == 4
    assert problem.n_evals == N + N * T + int(res.extra["ls_evals"].sum())
