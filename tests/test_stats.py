"""Pruebas de bfs.stats con DataFrames sintéticos (no dependen de results/)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from bfs.stats import (
    ALPHA,
    compare_all,
    holm,
    pvalue_matrix,
    summarize,
    summary_table,
    wilcoxon_paired,
)

RUNS = 30


def hacer_df(valores: dict[tuple[str, str], np.ndarray], metric: str = "best_fitness") -> pd.DataFrame:
    """Construye un runs.csv sintético: valores[(dataset, method)] = array de RUNS."""
    filas = []
    for (dataset, method), vals in valores.items():
        for r, v in enumerate(vals):
            filas.append({
                "dataset": dataset, "method": method, "run": r,
                "split_seed": r, "algo_seed": 1000 + r, metric: float(v),
            })
    return pd.DataFrame(filas)


# -- summarize ----------------------------------------------------------------

def test_summarize_media_mediana_y_desviacion():
    vals = np.arange(1.0, RUNS + 1)          # 1..30
    df = hacer_df({("d1", "m1"): vals})
    out = summarize(df, metrics=["best_fitness"])

    fila = out.loc[("d1", "m1")]
    assert fila[("best_fitness", "mean")] == pytest.approx(vals.mean())
    assert fila[("best_fitness", "median")] == pytest.approx(np.median(vals))
    assert fila[("best_fitness", "std")] == pytest.approx(vals.std(ddof=1))   # ddof = 1
    assert fila[("n", "")] == RUNS


def test_summarize_separa_por_dataset_y_metodo():
    df = hacer_df({
        ("d1", "m1"): np.zeros(RUNS), ("d1", "m2"): np.ones(RUNS),
        ("d2", "m1"): np.full(RUNS, 2.0),
    })
    out = summarize(df, metrics=["best_fitness"])
    assert len(out) == 3
    assert out.loc[("d1", "m2"), ("best_fitness", "mean")] == 1.0
    assert out.loc[("d2", "m1"), ("best_fitness", "mean")] == 2.0


def test_summarize_exige_que_la_metrica_exista():
    df = hacer_df({("d1", "m1"): np.zeros(RUNS)})
    with pytest.raises(KeyError):
        summarize(df, metrics=["no_existe"])


def test_summary_table_tiene_metodos_en_columnas():
    df = hacer_df({("d1", "m1"): np.zeros(RUNS), ("d1", "m2"): np.ones(RUNS)})
    tabla = summary_table(df, "best_fitness")
    assert list(tabla.columns) == ["m1", "m2"]
    assert "±" in tabla.loc["d1", "m1"]


# -- wilcoxon_paired ----------------------------------------------------------

def test_a_gana_cuando_su_fitness_es_menor():
    rng = np.random.default_rng(0)
    base = rng.random(RUNS)
    df = hacer_df({("d1", "a"): base, ("d1", "b"): base + 0.05})   # a mejor (menos fitness)
    out = wilcoxon_paired(df, "best_fitness", "a", "b")

    assert len(out) == 1
    fila = out.iloc[0]
    assert fila["dataset"] == "d1" and fila["n"] == RUNS
    assert fila["median_diff"] == pytest.approx(-0.05)
    assert fila["p_value"] < ALPHA
    assert fila["ganador"] == "a"


def test_b_gana_cuando_su_fitness_es_menor():
    rng = np.random.default_rng(1)
    base = rng.random(RUNS)
    df = hacer_df({("d1", "a"): base + 0.05, ("d1", "b"): base})
    out = wilcoxon_paired(df, "best_fitness", "a", "b")
    assert out.iloc[0]["ganador"] == "b"


def test_en_accuracy_mas_es_mejor():
    """La dirección se invierte: en accuracy gana quien tiene el valor más alto."""
    rng = np.random.default_rng(2)
    base = 80 + rng.random(RUNS)
    df = hacer_df({("d1", "a"): base + 5, ("d1", "b"): base}, metric="accuracy")
    out = wilcoxon_paired(df, "accuracy", "a", "b")
    assert out.iloc[0]["median_diff"] == pytest.approx(5.0)
    assert out.iloc[0]["ganador"] == "a"       # a tiene MÁS exactitud


def test_todas_las_diferencias_cero_es_empate():
    """Pasa en las bases de 9 variables: scipy no puede, regresamos p = 1."""
    vals = np.full(RUNS, 0.0286)
    df = hacer_df({("d1", "a"): vals, ("d1", "b"): vals.copy()})
    out = wilcoxon_paired(df, "best_fitness", "a", "b")
    fila = out.iloc[0]
    assert fila["p_value"] == 1.0
    assert fila["median_diff"] == 0.0
    assert fila["ganador"] == "empate"


def test_diferencia_minima_no_significativa_es_empate():
    rng = np.random.default_rng(3)
    a = rng.random(RUNS)
    b = a.copy()
    b[0] += 1e-6                                # una sola corrida difiere
    df = hacer_df({("d1", "a"): a, ("d1", "b"): b})
    out = wilcoxon_paired(df, "best_fitness", "a", "b")
    assert out.iloc[0]["p_value"] >= ALPHA
    assert out.iloc[0]["ganador"] == "empate"


def test_coincide_con_scipy():
    rng = np.random.default_rng(4)
    a, b = rng.random(RUNS), rng.random(RUNS)
    df = hacer_df({("d1", "a"): a, ("d1", "b"): b})
    out = wilcoxon_paired(df, "best_fitness", "a", "b").iloc[0]
    esperado = sps.wilcoxon(a, b)
    assert out["statistic"] == pytest.approx(esperado.statistic)
    assert out["p_value"] == pytest.approx(esperado.pvalue)


def test_el_pareo_es_por_run_no_por_orden_de_filas():
    """Revolver las filas no cambia el resultado: las parejas salen de `run`."""
    rng = np.random.default_rng(5)
    a, b = rng.random(RUNS), rng.random(RUNS)
    df = hacer_df({("d1", "a"): a, ("d1", "b"): b})
    revuelto = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    izq = wilcoxon_paired(df, "best_fitness", "a", "b").iloc[0]
    der = wilcoxon_paired(revuelto, "best_fitness", "a", "b").iloc[0]
    assert izq["p_value"] == pytest.approx(der["p_value"])
    assert izq["median_diff"] == pytest.approx(der["median_diff"])


def test_una_fila_por_dataset_ordenada():
    rng = np.random.default_rng(6)
    vals = {}
    for d in ("d2", "d1", "d3"):
        base = rng.random(RUNS)
        vals[(d, "a")] = base
        vals[(d, "b")] = base + 0.05
    out = wilcoxon_paired(hacer_df(vals), "best_fitness", "a", "b")
    assert list(out["dataset"]) == ["d1", "d2", "d3"]


def test_solo_usa_las_corridas_presentes_en_ambos():
    rng = np.random.default_rng(7)
    a = rng.random(RUNS)
    df = hacer_df({("d1", "a"): a, ("d1", "b"): a + 0.05})
    df = df[~((df["method"] == "b") & (df["run"] >= 20))]   # b solo tiene 20 corridas
    assert wilcoxon_paired(df, "best_fitness", "a", "b").iloc[0]["n"] == 20


def test_metodo_o_metrica_desconocidos():
    df = hacer_df({("d1", "a"): np.zeros(RUNS), ("d1", "b"): np.ones(RUNS)})
    with pytest.raises(KeyError):
        wilcoxon_paired(df, "best_fitness", "a", "no_existe")
    with pytest.raises(KeyError):
        wilcoxon_paired(df, "no_existe", "a", "b")


def test_metrica_sin_direccion_definida_avisa():
    df = hacer_df({("d1", "a"): np.zeros(RUNS), ("d1", "b"): np.ones(RUNS)}, metric="raro")
    with pytest.raises(ValueError, match="más o menos"):
        wilcoxon_paired(df, "raro", "a", "b")


# -- Holm ---------------------------------------------------------------------

def test_holm_valores_conocidos():
    # m = 4: p_i * (m - i) con monotonía acumulada
    np.testing.assert_allclose(holm([0.01, 0.02, 0.03, 0.04]), [0.04, 0.06, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.001, 0.9]), [0.002, 0.9])
    np.testing.assert_allclose(holm([0.5]), [0.5])


def test_holm_respeta_el_orden_original():
    np.testing.assert_allclose(holm([0.04, 0.01]), [0.04, 0.02])


def test_holm_acota_a_uno_y_nunca_reduce():
    p = np.array([0.6, 0.7, 0.2])
    aj = holm(p)
    assert (aj <= 1.0).all()
    assert (aj >= p - 1e-12).all()       # corregir solo puede subir el p-valor


def test_holm_es_mas_potente_que_bonferroni():
    p = np.array([0.01, 0.02, 0.03, 0.04])
    assert (holm(p) <= len(p) * p + 1e-12).all()


def test_holm_casos_borde():
    assert holm([]).size == 0
    np.testing.assert_allclose(holm([1.0, 1.0]), [1.0, 1.0])
    with pytest.raises(ValueError):
        holm([0.5, 1.5])
    with pytest.raises(ValueError):
        holm([0.5, np.nan])


# -- compare_all --------------------------------------------------------------

def test_compare_all_estructura_y_resultado():
    rng = np.random.default_rng(8)
    base = rng.random(RUNS)
    df = hacer_df({
        ("d1", "cbpso_miws"): base,
        ("d1", "cbpso_miws_ls"): base - 0.05,     # mejor que la referencia
        ("d1", "cbpso_miws_v"): base + 0.05,      # peor
    })
    out = compare_all(df, challengers=("cbpso_miws_ls", "cbpso_miws_v"),
                      metrics=("best_fitness",))

    assert set(out["method"]) == {"cbpso_miws_ls", "cbpso_miws_v"}
    assert (out["reference"] == "cbpso_miws").all()
    ls = out[out["method"] == "cbpso_miws_ls"].iloc[0]
    v = out[out["method"] == "cbpso_miws_v"].iloc[0]
    assert ls["resultado"] == "mejor"             # desde el punto de vista del retador
    assert v["resultado"] == "peor"
    assert "p_holm" in out.columns and "resultado_holm" in out.columns
    assert (out["p_holm"] >= out["p_value"] - 1e-12).all()


def test_compare_all_ignora_retadores_ausentes():
    df = hacer_df({("d1", "cbpso_miws"): np.zeros(RUNS), ("d1", "cbpso_miws_ls"): np.ones(RUNS)})
    out = compare_all(df, challengers=("cbpso_miws_ls", "no_corrido"),
                      metrics=("best_fitness",))
    assert set(out["method"]) == {"cbpso_miws_ls"}


def test_compare_all_sin_correccion():
    df = hacer_df({("d1", "cbpso_miws"): np.zeros(RUNS), ("d1", "cbpso_miws_ls"): np.ones(RUNS)})
    out = compare_all(df, challengers=("cbpso_miws_ls",), metrics=("best_fitness",),
                      correction=None)
    assert "p_holm" not in out.columns
    with pytest.raises(ValueError):
        compare_all(df, challengers=("cbpso_miws_ls",), metrics=("best_fitness",),
                    correction="bonferroni")


def test_pvalue_matrix():
    rng = np.random.default_rng(9)
    vals = {}
    for d in ("d1", "d2"):
        base = rng.random(RUNS)
        vals[(d, "cbpso_miws")] = base
        vals[(d, "cbpso_miws_ls")] = base - 0.05
    out = compare_all(hacer_df(vals), challengers=("cbpso_miws_ls",), metrics=("best_fitness",))
    m = pvalue_matrix(out, "best_fitness")
    assert list(m.index) == ["d1", "d2"]
    assert list(m.columns) == ["cbpso_miws_ls"]
    assert (m.to_numpy() < ALPHA).all()


# -- robustez al ruido de punto flotante -------------------------------------

def test_ruido_de_punto_flotante_no_se_rankea():
    """Un valor releído del CSV difiere ~1e-17: es la misma solución, no una mejora.

    Caso mixto (lo peligroso): algunas diferencias reales y otras de puro ruido.
    Sin el umbral, el ruido entra al ranking con los rangos más chicos y corre
    el estadístico.
    """
    rng = np.random.default_rng(10)
    a = rng.random(RUNS)
    b = a.copy()
    # Ruido de 1 ULP con signos mezclados (el redondeo del CSV va en cualquier dirección)
    b[:20:2] = np.nextafter(b[:20:2], 1.0)
    b[1:20:2] = np.nextafter(b[1:20:2], 0.0)
    b[20:] += 0.05                       # 10 diferencias reales
    df = hacer_df({("d1", "a"): a, ("d1", "b"): b})

    con_umbral = wilcoxon_paired(df, "best_fitness", "a", "b").iloc[0]
    sin_umbral = wilcoxon_paired(df, "best_fitness", "a", "b", tol=0.0).iloc[0]

    # Con umbral: la prueba es exactamente la de las 10 diferencias reales
    esperado = sps.wilcoxon(a[20:] - b[20:])
    assert con_umbral["statistic"] == pytest.approx(esperado.statistic)
    assert con_umbral["p_value"] == pytest.approx(esperado.pvalue)
    # Sin umbral, el ruido ocupa los rangos más chicos y corre el resultado
    assert sin_umbral["statistic"] != con_umbral["statistic"]
    assert sin_umbral["p_value"] != pytest.approx(con_umbral["p_value"])


def test_ruido_puro_sigue_siendo_empate():
    """30 diferencias de ~1e-17 del mismo signo no deben dar un p-valor diminuto."""
    rng = np.random.default_rng(11)
    a = rng.random(RUNS)
    b = np.nextafter(a, 1.0)             # sesgo sistemático de 1 ULP, todo ruido
    df = hacer_df({("d1", "a"): a, ("d1", "b"): b})

    out = wilcoxon_paired(df, "best_fitness", "a", "b").iloc[0]
    assert out["p_value"] == 1.0
    assert out["ganador"] == "empate"
    # Sin umbral, la prueba declararía una diferencia significativa inexistente
    assert wilcoxon_paired(df, "best_fitness", "a", "b", tol=0.0).iloc[0]["p_value"] < ALPHA


def test_el_umbral_no_toca_diferencias_reales():
    """Una diferencia de 1e-6 (real, aunque chica) sí se cuenta."""
    rng = np.random.default_rng(12)
    a = rng.random(RUNS)
    b = a + 1e-6
    out = wilcoxon_paired(hacer_df({("d1", "a"): a, ("d1", "b"): b}), "best_fitness", "a", "b")
    assert out.iloc[0]["p_value"] < ALPHA
    assert out.iloc[0]["ganador"] == "a"
