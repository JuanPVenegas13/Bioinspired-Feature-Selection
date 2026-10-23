"""Pruebas de la comparación réplica vs. artículo (no requieren correr algoritmos)."""

import numpy as np
import pandas as pd
import pytest

from bfs.data import DATASETS
from bfs.replication import (
    compare_with_paper,
    load_paper_results,
    paper_value,
    side_by_side,
    wilcoxon_one_sample,
)


def test_paper_table_covers_our_datasets():
    paper = load_paper_results()
    for dataset in DATASETS:
        for method in ("bpso", "cbpso_miws"):
            assert paper_value(paper, dataset, method, "accuracy") > 0


def test_paper_values_transcribed():
    paper = load_paper_results()
    # Tabla 3 del artículo
    assert paper_value(paper, "ionosphere", "cbpso_miws", "accuracy") == 91.36
    assert paper_value(paper, "lung_cancer", "bpso", "n_features") == 20.10
    # glass_id se compara contra la fila de Glass
    assert paper_value(paper, "glass_id", "cbpso_miws", "accuracy") == 98.37


def test_one_sample_wilcoxon():
    rng = np.random.default_rng(0)
    igual = 90 + rng.normal(0, 1, 30)
    assert wilcoxon_one_sample(igual, 90.0)[1] > 0.05
    assert wilcoxon_one_sample(igual + 3, 90.0)[1] < 0.001
    assert wilcoxon_one_sample(np.full(30, 95.0), 95.0) == (0.0, 1.0)  # todas iguales


def _fake_runs(acc_cbpso: float, acc_bpso: float) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    filas = []
    for method, acc in [("cbpso_miws", acc_cbpso), ("bpso", acc_bpso)]:
        for r in range(30):
            filas.append({"dataset": "ionosphere", "method": method, "run": r,
                          "accuracy": acc + rng.normal(0, 1), "best_fitness": 0.05,
                          "mean_fitness": 0.06, "n_features": 10})
    return pd.DataFrame(filas)


def test_compare_with_paper_directions():
    # Artículo: CBPSO-MIWS 91.36, BPSO 88.00 en Ionosphere
    out = compare_with_paper(_fake_runs(95.0, 88.0), metrics=("accuracy",))
    res = out.set_index("method")["resultado"]
    assert res["cbpso_miws"] == "mayor"
    assert res["bpso"] == "igual"
    assert set(out.columns) >= {"ours_mean", "ours_median", "ours_std", "paper", "p_value", "p_holm"}


def test_side_by_side_shape():
    tabla = side_by_side(_fake_runs(95.0, 88.0))
    assert list(tabla.columns) == ["BPSO artículo", "BPSO réplica",
                                   "CBPSO-MIWS artículo", "CBPSO-MIWS réplica"]
    assert tabla.loc["ionosphere", "CBPSO-MIWS artículo"] == pytest.approx(91.36)
