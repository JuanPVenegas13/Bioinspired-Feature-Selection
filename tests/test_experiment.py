"""Pruebas del runner de experimentos (contrato con la etapa 3, sección 3).

Se usa ``breast_cancer_coimbra`` (116 instancias, 9 variables) con T = 5 para
que sea rápido, y ``tmp_path`` de pytest para no tocar ``results/``.
"""

import numpy as np
import pandas as pd
import pytest

from bfs.data import RAW_DIR
from bfs.experiment import (
    ALGO_SEED_BASE,
    METHODS,
    RUN_COLUMNS,
    curve_key,
    evals_curve,
    run_experiment,
    run_one,
)

pytestmark = pytest.mark.skipif(
    not RAW_DIR.exists() or not any(RAW_DIR.iterdir()),
    reason="faltan datos: corre `uv run python scripts/download_data.py`",
)

BASE = "breast_cancer_coimbra"
METODOS = ["cbpso_miws", "cbpso_miws_ls"]
RUNS, T = 2, 5


@pytest.fixture(scope="module")
def experimento(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    df = run_experiment(
        datasets=[BASE], methods=METODOS, runs=RUNS, T=T,
        out_dir=out, workers=1, verbose=False,
    )
    return df, out


# -- contrato de runs.csv -----------------------------------------------------

def test_columnas_exactas_y_orden(experimento):
    df, out = experimento
    assert list(df.columns) == RUN_COLUMNS
    guardado = pd.read_csv(out / "runs.csv")
    assert list(guardado.columns) == RUN_COLUMNS
    # ordenado por dataset, method, run
    esperado = guardado.sort_values(["dataset", "method", "run"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(guardado, esperado)


def test_una_fila_por_combinacion(experimento):
    df, _ = experimento
    assert len(df) == len(METODOS) * RUNS
    assert not df.duplicated(["dataset", "method", "run"]).any()
    assert sorted(df["method"].unique()) == sorted(METODOS)
    assert sorted(df["run"].unique()) == list(range(RUNS))


def test_semillas_pareadas(experimento):
    """La corrida r usa la misma partición y el mismo generador en todo método."""
    df, _ = experimento
    assert (df["split_seed"] == df["run"]).all()
    assert (df["algo_seed"] == ALGO_SEED_BASE + df["run"]).all()


def test_ecuaciones_13_a_16_sobre_la_curva(experimento):
    """best/worst/mean/std salen de la curva de convergencia de cada corrida."""
    df, out = experimento
    with np.load(out / "curves.npz") as z:
        for _, fila in df.iterrows():
            curva = z[curve_key(fila["dataset"], fila["method"])][int(fila["run"])]
            assert fila["best_fitness"] == pytest.approx(curva.min())    # ec. 13
            assert fila["worst_fitness"] == pytest.approx(curva.max())   # ec. 14
            assert fila["mean_fitness"] == pytest.approx(curva.mean())   # ec. 15
            assert fila["std_fitness"] == pytest.approx(curva.std(ddof=0))  # ec. 16


def test_rangos_razonables(experimento):
    df, _ = experimento
    assert df["best_fitness"].between(0, 1).all()
    assert df["accuracy"].between(0, 100).all()
    assert df["n_features"].between(1, 9).all()
    assert (df["n_evals"] > 0).all()
    assert (df["time_s"] > 0).all()


def test_columna_selected_coincide_con_n_features(experimento):
    df, _ = experimento
    for _, fila in df.iterrows():
        indices = [int(i) for i in str(fila["selected"]).split(";") if i != ""]
        assert len(indices) == fila["n_features"]
        assert indices == sorted(set(indices))


def test_la_busqueda_local_gasta_mas_evaluaciones(experimento):
    df, _ = experimento
    por_metodo = df.groupby("method")["n_evals"].mean()
    assert por_metodo["cbpso_miws_ls"] > por_metodo["cbpso_miws"]


# -- contrato de curves.npz ---------------------------------------------------

def test_curvas_con_forma_n_runs_por_T(experimento):
    _, out = experimento
    with np.load(out / "curves.npz") as z:
        assert sorted(z.files) == sorted(curve_key(BASE, m) for m in METODOS)
        for key in z.files:
            assert z[key].shape == (RUNS, T)
            # la curva nunca empeora (contrato de Result)
            assert np.all(np.diff(z[key], axis=1) <= 1e-12)


# -- reproducibilidad y reanudación ------------------------------------------

def test_dos_corridas_dan_los_mismos_numeros(experimento, tmp_path):
    df, _ = experimento
    otra = run_experiment(
        datasets=[BASE], methods=METODOS, runs=RUNS, T=T,
        out_dir=tmp_path / "otra", workers=1, verbose=False,
    )
    columnas = [c for c in RUN_COLUMNS if c != "time_s"]  # el tiempo sí varía
    pd.testing.assert_frame_equal(df[columnas], otra[columnas])


def test_paralelo_da_lo_mismo_que_secuencial(experimento, tmp_path):
    """El resultado no depende del orden ni del número de procesos."""
    df, _ = experimento
    paralelo = run_experiment(
        datasets=[BASE], methods=METODOS, runs=RUNS, T=T,
        out_dir=tmp_path / "par", workers=2, verbose=False,
    )
    columnas = [c for c in RUN_COLUMNS if c != "time_s"]
    pd.testing.assert_frame_equal(df[columnas], paralelo[columnas])


def test_run_one_coincide_con_el_runner(experimento):
    df, out = experimento
    fila, curva, _ = run_one(BASE, "cbpso_miws_ls", run=1, T=T)
    guardada = df[(df["method"] == "cbpso_miws_ls") & (df["run"] == 1)].iloc[0]
    for col in RUN_COLUMNS:
        if col == "time_s":
            continue
        assert fila[col] == pytest.approx(guardada[col]) if isinstance(
            fila[col], float
        ) else fila[col] == guardada[col]
    with np.load(out / "curves.npz") as z:
        np.testing.assert_allclose(curva, z[curve_key(BASE, "cbpso_miws_ls")][1])


def test_es_reanudable(tmp_path):
    """Lo que ya está en runs.csv no se repite, y lo que falta se agrega."""
    out = tmp_path / "res"
    primera = run_experiment(
        datasets=[BASE], methods=["cbpso_miws"], runs=RUNS, T=T,
        out_dir=out, workers=1, verbose=False,
    )
    assert len(primera) == RUNS
    tiempos = dict(zip(primera["run"], primera["time_s"]))

    # Se piden las mismas y una configuración más: solo corre las nuevas.
    segunda = run_experiment(
        datasets=[BASE], methods=["cbpso_miws", "cbpso_miws_ls"], runs=RUNS, T=T,
        out_dir=out, workers=1, verbose=False,
    )
    assert len(segunda) == 2 * RUNS
    ya_estaban = segunda[segunda["method"] == "cbpso_miws"]
    # No se volvieron a correr: conservan el tiempo medido la primera vez.
    # (approx porque el CSV redondea al escribir; una corrida nueva daría otro tiempo)
    for _, fila in ya_estaban.iterrows():
        assert fila["time_s"] == pytest.approx(tiempos[fila["run"]], rel=1e-9)

    with np.load(out / "curves.npz") as z:
        assert sorted(z.files) == sorted(
            curve_key(BASE, m) for m in ("cbpso_miws", "cbpso_miws_ls")
        )
        for key in z.files:
            assert z[key].shape == (RUNS, T)


def test_reanudar_cuando_ya_esta_todo(experimento, tmp_path):
    """Pedir lo mismo otra vez no cambia nada."""
    out = tmp_path / "completo"
    a = run_experiment(datasets=[BASE], methods=["cbpso_miws"], runs=RUNS, T=T,
                       out_dir=out, workers=1, verbose=False)
    b = run_experiment(datasets=[BASE], methods=["cbpso_miws"], runs=RUNS, T=T,
                       out_dir=out, workers=1, verbose=False)
    pd.testing.assert_frame_equal(a, b)


# -- parámetros de METHODS ----------------------------------------------------

def test_T_del_metodo_sobrescribe_el_general(tmp_path):
    """cbpso_miws_T150: el control de presupuesto usa su propio T."""
    assert METHODS["cbpso_miws_T150"] == ("cbpso_miws", {"T": 150})
    fila, curva, _ = run_one(BASE, "cbpso_miws_T150", run=0, T=10)
    assert len(curva) == 150                     # ganó el T del diccionario
    base, curva_base, _ = run_one(BASE, "cbpso_miws", run=0, T=10)
    assert len(curva_base) == 10
    assert fila["n_evals"] > base["n_evals"]     # más iteraciones, más evaluaciones


def test_methods_apunta_a_algoritmos_registrados():
    from bfs.algorithms import ALGORITHMS

    for nombre, (algo, params) in METHODS.items():
        assert algo in ALGORITHMS, f"{nombre} apunta a {algo!r}, que no está registrado"
        assert isinstance(params, dict)


def test_run_one_rechaza_claves_desconocidas():
    with pytest.raises(KeyError):
        run_one("no_existe", "cbpso_miws", run=0)
    with pytest.raises(KeyError):
        run_one(BASE, "no_existe", run=0)


def test_run_experiment_rechaza_claves_desconocidas(tmp_path):
    with pytest.raises(KeyError):
        run_experiment(datasets=["no_existe"], runs=1, T=T, out_dir=tmp_path, verbose=False)
    with pytest.raises(KeyError):
        run_experiment(methods=["no_existe"], runs=1, T=T, out_dir=tmp_path, verbose=False)


# -- eje de evaluaciones (evals.npz) -----------------------------------------

def test_evals_curve_modelo_de_costo():
    """El PSO gasta N al inicializar y N por iteración: N + N t."""
    np.testing.assert_array_equal(evals_curve(10, 3), [20, 30, 40])
    # la búsqueda local agrega su gasto acumulado
    np.testing.assert_array_equal(evals_curve(10, 3, [5, 0, 2]), [25, 35, 47])


@pytest.mark.parametrize("method", sorted(METHODS))
def test_la_curva_de_evaluaciones_cierra_con_n_evals(method):
    """Cuadratura del costo: el eje termina exactamente en problem.n_evals."""
    fila, curva, evals = run_one(BASE, method, run=0, T=T)
    assert len(evals) == len(curva)
    assert int(evals[-1]) == fila["n_evals"]
    assert np.all(np.diff(evals) > 0)            # estrictamente creciente


def test_evals_npz_mismas_claves_y_formas(experimento):
    _, out = experimento
    with np.load(out / "curves.npz") as curvas, np.load(out / "evals.npz") as evals:
        assert sorted(curvas.files) == sorted(evals.files)
        for key in curvas.files:
            assert evals[key].shape == curvas[key].shape


def test_evals_npz_coincide_con_la_columna_n_evals(experimento):
    df, out = experimento
    with np.load(out / "evals.npz") as z:
        for _, fila in df.iterrows():
            ultimo = z[curve_key(fila["dataset"], fila["method"])][int(fila["run"])][-1]
            assert int(ultimo) == fila["n_evals"]


def test_curves_npz_conserva_el_contrato_de_la_seccion_3(experimento):
    """evals.npz es ADICIONAL: curves.npz sigue teniendo solo dataset__method."""
    _, out = experimento
    with np.load(out / "curves.npz") as z:
        for key in z.files:
            dataset, sep, method = key.partition("__")
            assert sep == "__" and dataset == BASE and method in METODOS


def test_evals_se_reanuda_igual_que_las_curvas(tmp_path):
    out = tmp_path / "res"
    run_experiment(datasets=[BASE], methods=["cbpso_miws"], runs=RUNS, T=T,
                   out_dir=out, workers=1, verbose=False)
    with np.load(out / "evals.npz") as z:
        antes = {k: z[k].copy() for k in z.files}

    df = run_experiment(datasets=[BASE], methods=METODOS, runs=RUNS, T=T,
                        out_dir=out, workers=1, verbose=False)
    with np.load(out / "evals.npz") as z:
        assert sorted(z.files) == sorted(curve_key(BASE, m) for m in METODOS)
        # lo que ya estaba no se perdió ni cambió
        for k, v in antes.items():
            np.testing.assert_array_equal(z[k], v)
        for _, fila in df.iterrows():
            ultimo = z[curve_key(fila["dataset"], fila["method"])][int(fila["run"])][-1]
            assert int(ultimo) == fila["n_evals"]
