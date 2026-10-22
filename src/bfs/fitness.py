"""Función de fitness del enfoque wrapper (Too et al., 2019, ecuaciones 11-12).

    Fitness = alpha * |S| / |T| + (1 - alpha) * Error

    |S|   : número de variables seleccionadas
    |T|   : número total de variables
    Error : tasa de error de un KNN (k = 5, distancia euclidiana) entrenado
            con las variables seleccionadas y evaluado en el 20 % de prueba
    alpha : 0.01 en el artículo (el error pesa 99 %, el tamaño 1 %)

El problema se MINIMIZA. Una solución es una máscara binaria de longitud |T|
(1 = variable seleccionada, 0 = descartada).
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from bfs.data import Split

EMPTY_SUBSET_FITNESS = 1.0
"""Fitness asignado a una máscara sin variables (peor que cualquier solución real).
El MATLAB del autor usa inf; usamos 1.0 para no romper promedios."""


class FeatureSelectionProblem:
    """Evalúa máscaras binarias sobre una partición 80/20 fija.

    Uso::

        problem = FeatureSelectionProblem(holdout_split(ds, seed=0))
        f = problem(mask)            # fitness (ecuación 11)
        acc = problem.accuracy(mask) # exactitud en % (ecuación 17)

    Las evaluaciones se guardan en caché: el KNN es determinista, así que una
    misma máscara siempre da el mismo fitness. ``n_evals`` cuenta las llamadas
    hechas por el algoritmo (con o sin caché), útil para comparar costo.
    """

    def __init__(self, split: Split, k: int = 5, alpha: float = 0.01):
        self.split = split
        self.k = k
        self.alpha = alpha
        self.dim = split.X_train.shape[1]
        self.n_evals = 0
        self._cache: dict[bytes, tuple[float, float]] = {}

    # -- núcleo ---------------------------------------------------------------
    def _evaluate(self, mask: np.ndarray) -> tuple[float, float]:
        """Regresa (fitness, error) de una máscara, usando caché."""
        mask = np.asarray(mask, dtype=bool)
        key = np.packbits(mask).tobytes()
        if key in self._cache:
            return self._cache[key]

        n_sel = int(mask.sum())
        if n_sel == 0:
            out = (EMPTY_SUBSET_FITNESS, 1.0)
        else:
            knn = KNeighborsClassifier(n_neighbors=self.k)  # euclidiana por defecto
            knn.fit(self.split.X_train[:, mask], self.split.y_train)
            pred = knn.predict(self.split.X_test[:, mask])
            error = float(np.mean(pred != self.split.y_test))  # ecuación 12
            fitness = self.alpha * n_sel / self.dim + (1 - self.alpha) * error
            out = (fitness, error)

        self._cache[key] = out
        return out

    # -- API ------------------------------------------------------------------
    def __call__(self, mask: np.ndarray) -> float:
        self.n_evals += 1
        return self._evaluate(mask)[0]

    def error(self, mask: np.ndarray) -> float:
        return self._evaluate(mask)[1]

    def accuracy(self, mask: np.ndarray) -> float:
        """Exactitud en porcentaje (ecuación 17)."""
        return 100.0 * (1.0 - self.error(mask))
