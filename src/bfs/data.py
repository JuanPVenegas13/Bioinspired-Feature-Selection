"""Carga y preprocesamiento de las bases de UCI usadas en la réplica.

Protocolo del artículo (Too et al., 2019, sección 4.1):
  * las variables se normalizan al rango [0, 1];
  * en cada evaluación de fitness los datos se parten al azar en
    80 % entrenamiento / 20 % prueba.

Decisiones propias (el artículo no las especifica):
  * valores faltantes ('?') se imputan con la mediana de la columna;
  * la partición 80/20 es estratificada por clase (igual que
    ``cvpartition(label, 'HoldOut', 0.2)`` de MATLAB, que usa el autor);
  * las etiquetas se recodifican a enteros 0..k-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass(frozen=True)
class Dataset:
    """Base de datos lista para usarse: X normalizada en [0, 1], y en 0..k-1."""

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]

    @property
    def n_instances(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return len(np.unique(self.y))

    def __repr__(self) -> str:
        return (
            f"Dataset({self.name!r}, instancias={self.n_instances}, "
            f"variables={self.n_features}, clases={self.n_classes})"
        )


@dataclass(frozen=True)
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


# --------------------------------------------------------------------------
# Lectores de cada archivo original. Cada uno regresa (X crudo, y crudo, nombres)
# --------------------------------------------------------------------------

def _read_breast_cancer_wisconsin(raw_dir: Path):
    # id, 9 variables enteras (1-10), clase (2 = benigno, 4 = maligno)
    df = pd.read_csv(raw_dir / "breast-cancer-wisconsin.data", header=None, na_values="?")
    names = [
        "clump_thickness", "cell_size", "cell_shape", "marginal_adhesion",
        "epithelial_size", "bare_nuclei", "bland_chromatin",
        "normal_nucleoli", "mitoses",
    ]
    return df.iloc[:, 1:10], df.iloc[:, 10], names


def _read_glass(raw_dir: Path, include_id: bool):
    # id, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe, tipo (1, 2, 3, 5, 6, 7)
    df = pd.read_csv(raw_dir / "glass.data", header=None)
    names = ["id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
    start = 0 if include_id else 1
    return df.iloc[:, start:10], df.iloc[:, 10], names[start:]


def _read_ionosphere(raw_dir: Path):
    # 34 variables continuas, clase 'g' / 'b'
    df = pd.read_csv(raw_dir / "ionosphere.data", header=None)
    return df.iloc[:, :34], df.iloc[:, 34], [f"a{i + 1:02d}" for i in range(34)]


def _read_libras(raw_dir: Path):
    # 90 variables (45 pares de coordenadas x, y), clase 1..15
    df = pd.read_csv(raw_dir / "movement_libras.data", header=None)
    names = [f"{c}{i + 1:02d}" for i in range(45) for c in ("x", "y")]
    return df.iloc[:, :90], df.iloc[:, 90], names


def _read_breast_cancer_coimbra(raw_dir: Path):
    # encabezado: Age, BMI, ..., MCP.1, Classification (1 = sano, 2 = paciente)
    df = pd.read_csv(raw_dir / "dataR2.csv")
    X = df.drop(columns="Classification")
    return X, df["Classification"], list(X.columns)


def _read_lung_cancer(raw_dir: Path):
    # clase (1..3) en la PRIMERA columna, luego 56 variables ordinales (0-3)
    df = pd.read_csv(raw_dir / "lung-cancer.data", header=None, na_values="?")
    return df.iloc[:, 1:], df.iloc[:, 0], [f"a{i + 1:02d}" for i in range(56)]


# Número de la base según la Tabla 1 del artículo -> lector
DATASETS: dict[str, tuple[int, Callable]] = {
    "breast_cancer_wisconsin": (1, _read_breast_cancer_wisconsin),
    "glass": (3, lambda d: _read_glass(d, include_id=False)),
    "glass_id": (3, lambda d: _read_glass(d, include_id=True)),
    "ionosphere": (4, _read_ionosphere),
    "libras": (5, _read_libras),
    "breast_cancer_coimbra": (7, _read_breast_cancer_coimbra),
    "lung_cancer": (8, _read_lung_cancer),
}
"""Bases disponibles.

``glass`` usa las 9 variables químicas. ``glass_id`` agrega la columna de ID
(10 variables, como reporta la Tabla 1 del artículo). Ojo: el ID está ordenado
por clase, así que filtra la etiqueta; se incluye solo para comparar.
"""


# --------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------

def minmax_normalize(X: np.ndarray) -> np.ndarray:
    """Escala cada columna a [0, 1]. Columnas constantes quedan en 0."""
    xmin = X.min(axis=0)
    rng = X.max(axis=0) - xmin
    rng[rng == 0] = 1.0
    return (X - xmin) / rng


def load_dataset(name: str, raw_dir: Path | str | None = None) -> Dataset:
    """Lee, imputa, normaliza y recodifica una base. Ver ``DATASETS``."""
    if name not in DATASETS:
        raise KeyError(f"Base desconocida {name!r}. Opciones: {sorted(DATASETS)}")
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DIR
    _, reader = DATASETS[name]
    X_raw, y_raw, names = reader(raw_dir)

    X_df = X_raw.astype(float)
    X_df = X_df.fillna(X_df.median())  # imputación por mediana
    X = minmax_normalize(X_df.to_numpy())

    _, y = np.unique(y_raw.to_numpy(), return_inverse=True)  # -> 0..k-1
    return Dataset(name=name, X=X, y=y.astype(int), feature_names=list(names))


def holdout_split(ds: Dataset, seed: int, test_size: float = 0.2) -> Split:
    """Partición aleatoria estratificada (por defecto 80/20, como el artículo)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=test_size, stratify=ds.y, random_state=seed
    )
    return Split(X_tr, X_te, y_tr, y_te)
