import math

import numpy as np
import pytest

from bfs.data import DATASETS, RAW_DIR, holdout_split, load_dataset

# (instancias, variables, clases) según la Tabla 1 del artículo
EXPECTED = {
    "breast_cancer_wisconsin": (699, 9, 2),
    "glass": (214, 9, 6),
    "glass_id": (214, 10, 6),
    "ionosphere": (351, 34, 2),
    "libras": (360, 90, 15),
    "breast_cancer_coimbra": (116, 9, 2),
    "lung_cancer": (32, 56, 3),
}

pytestmark = pytest.mark.skipif(
    not RAW_DIR.exists() or not any(RAW_DIR.iterdir()),
    reason="faltan datos: corre `uv run python scripts/download_data.py`",
)


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_shape_and_normalization(name):
    ds = load_dataset(name)
    assert (ds.n_instances, ds.n_features, ds.n_classes) == EXPECTED[name]
    assert not np.isnan(ds.X).any()
    assert ds.X.min() >= 0.0 and ds.X.max() <= 1.0
    assert set(np.unique(ds.y)) == set(range(ds.n_classes))


def test_split_is_80_20_stratified_and_reproducible():
    ds = load_dataset("ionosphere")
    a = holdout_split(ds, seed=7)
    b = holdout_split(ds, seed=7)
    assert len(a.y_test) == math.ceil(0.2 * ds.n_instances)  # sklearn redondea hacia arriba
    np.testing.assert_array_equal(a.y_test, b.y_test)
    # proporción de clases preservada (±5 %)
    assert abs(a.y_test.mean() - ds.y.mean()) < 0.05
