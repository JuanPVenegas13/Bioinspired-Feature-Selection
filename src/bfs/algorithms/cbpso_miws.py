"""CBPSO-MIWS: Co-evolution BPSO con Multiple Inertia Weight Strategy.

Algoritmo propuesto en Too et al. (2019), sección 3 / Algoritmo 1:

  1. La población de N partículas se divide en ``ns`` especies
     (sub-poblaciones). Cada especie tiene su propio mejor global ``gbest_n``.
  2. En cada iteración, CADA especie sortea uno de 4 esquemas de inercia
     (ec. 10, probabilidad 1/4 cada uno) y calcula su ``w``:
        IWS 1 (ec. 6): w = wmax - (wmax - wmin) t / Tmax         (lineal)
        IWS 2 (ec. 7): w = 0.5 + r3 / 2                          (aleatorio)
        IWS 3 (ec. 8): w = ((Tmax - t)^p / Tmax^p)(wmax - wmin) + wmin   (no lineal)
        IWS 4 (ec. 9): w = w0                                    (constante)
  3. Cada partícula se mueve con BPSO estándar (ec. 1-3), pero guiada por
     su pbest y por el gbest DE SU ESPECIE.
  4. Al final de cada iteración se actualiza el mejor global de todas las
     especies, Gbest, que es la salida del algoritmo.

Nota de interpretación: con N = 10 y ns = 3 la división no es exacta;
usamos especies de tamaño 4, 3 y 3 (``np.array_split``).

Parámetros para la ablación:
  * ``ns = 1`` -> sin co-evolución (solo inercia múltiple)
  * ``schemes = (1,)`` -> sin inercia múltiple (solo co-evolución, IWS 1 fijo)
"""

from __future__ import annotations

import numpy as np

from bfs.algorithms.base import FitnessFn, Result, random_binary_population, register
from bfs.algorithms.bpso import update_particle


def inertia_weight(
    scheme: int, t: int, T: int, rng: np.random.Generator,
    wmax: float, wmin: float, w0: float, p: float,
) -> float:
    """Ecuaciones 6-9."""
    if scheme == 1:
        return wmax - (wmax - wmin) * (t / T)
    if scheme == 2:
        return 0.5 + 0.5 * rng.random()
    if scheme == 3:
        return ((T - t) ** p / T**p) * (wmax - wmin) + wmin
    if scheme == 4:
        return w0
    raise ValueError(f"Esquema de inercia desconocido: {scheme}")


@register("cbpso_miws")
def run(
    fitness: FitnessFn, dim: int, N: int, T: int, rng: np.random.Generator,
    ns: int = 3, c1: float = 2.0, c2: float = 2.0, wmax: float = 0.9,
    wmin: float = 0.4, w0: float = 0.9, p: float = 1.2, vmax: float = 6.0,
    schemes: tuple[int, ...] = (1, 2, 3, 4),
) -> Result:
    """CBPSO-MIWS (Algoritmo 1). Valores por defecto = Tabla 2 del artículo."""
    if not 1 <= ns <= N:
        raise ValueError(f"ns debe estar entre 1 y N={N}, recibí {ns}")

    # Pasos 1-5: inicializar, dividir en especies, evaluar, pbest / gbest_n / Gbest
    X = random_binary_population(N, dim, rng)
    V = np.zeros((N, dim))
    fit = np.array([fitness(x) for x in X])
    pbest, fit_p = X.copy(), fit.copy()

    species = np.array_split(np.arange(N), ns)  # índices de cada especie
    gbest_n, fit_gn = [], []
    for idx in species:
        j = idx[np.argmin(fit_p[idx])]
        gbest_n.append(pbest[j].copy())
        fit_gn.append(float(fit_p[j]))
    k = int(np.argmin(fit_gn))
    Gbest, fit_G = gbest_n[k].copy(), fit_gn[k]

    curve = np.empty(T)
    iws_history = np.empty((T, ns), dtype=np.int8)

    for t in range(1, T + 1):                       # paso 6
        for n, idx in enumerate(species):           # paso 7
            scheme = int(rng.choice(schemes))       # paso 8, ec. 10
            iws_history[t - 1, n] = scheme
            w = inertia_weight(scheme, t, T, rng, wmax, wmin, w0, p)  # paso 9

            for i in idx:                           # pasos 10-18
                X[i], V[i] = update_particle(
                    X[i], V[i], pbest[i], gbest_n[n], w, c1, c2, vmax, rng
                )
                f = fitness(X[i])
                if f < fit_p[i]:
                    pbest[i], fit_p[i] = X[i].copy(), f
                    if f < fit_gn[n]:
                        gbest_n[n], fit_gn[n] = X[i].copy(), f

        k = int(np.argmin(fit_gn))                  # paso 20
        if fit_gn[k] < fit_G:
            Gbest, fit_G = gbest_n[k].copy(), fit_gn[k]
        curve[t - 1] = fit_G

    return Result(
        mask=Gbest.astype(bool), best_fitness=fit_G, curve=curve,
        extra={"iws_history": iws_history, "species_sizes": [len(s) for s in species]},
    )
