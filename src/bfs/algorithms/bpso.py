"""Binary Particle Swarm Optimization (BPSO), sección 2 del artículo.

Traducción directa de ``jBPSO.m`` del autor
(github.com/JingweiToo/Binary-Particle-Swarm-Optimization-for-Feature-Selection):

  * velocidad (ec. 1):  v = w v + c1 r1 (pbest - x) + c2 r2 (gbest - x)
  * acotada en [-vmax, vmax] = [-6, 6]
  * probabilidad (ec. 2): S(v) = 1 / (1 + e^-v)
  * posición (ec. 3):   x = 1 si rand < S(v), si no 0
  * inercia lineal decreciente de wmax = 0.9 a wmin = 0.4 (Tabla 2)
  * pbest y gbest se actualizan al evaluar cada partícula (ec. 4-5)
"""

from __future__ import annotations

import numpy as np

from bfs.algorithms.base import FitnessFn, Result, random_binary_population, register, sigmoid


def update_particle(
    x: np.ndarray, v: np.ndarray, pbest: np.ndarray, gbest: np.ndarray,
    w: float, c1: float, c2: float, vmax: float, rng: np.random.Generator,
    transfer: str = "s",
) -> tuple[np.ndarray, np.ndarray]:
    """Ecuación 1 + función de transferencia para UNA partícula.

    Regresa (x_nueva, v_nueva). Se reutiliza en CBPSO-MIWS, que solo cambia
    de dónde salen ``w`` y ``gbest``.

    transfer = "s" (artículo, ec. 2-3): el bit nuevo es 1 con probabilidad S(v).
    transfer = "v" (mejora 1, ver ``v_shaped``): el bit se VOLTEA con
                   probabilidad V(v); si no, conserva su valor.
    """
    dim = x.size
    r1 = rng.random(dim)
    r2 = rng.random(dim)
    v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
    v = np.clip(v, -vmax, vmax)
    if transfer == "s":
        x = (rng.random(dim) < sigmoid(v)).astype(np.int8)
    elif transfer == "v":
        flip = rng.random(dim) < v_shaped(v)
        x = np.where(flip, 1 - x, x).astype(np.int8)
    else:
        raise ValueError(f"transfer debe ser 's' o 'v', recibí {transfer!r}")
    return x, v


def v_shaped(v: np.ndarray) -> np.ndarray:
    """Función de transferencia en V: V(v) = |tanh(v)|.

    Mirjalili, S. y Lewis, A. (2013). S-shaped versus V-shaped transfer
    functions for binary Particle Swarm Optimization. Swarm and Evolutionary
    Computation, 9, 1-14. https://doi.org/10.1016/j.swevo.2012.09.002

    Diferencia clave con la sigmoide: con v = 0 la probabilidad de cambio es 0
    (la partícula se queda donde está) y crece con |v| en ambas direcciones.
    Con la sigmoide, v = 0 da probabilidad 0.5, es decir, bits al azar aun
    cuando la partícula ya convergió.
    """
    return np.abs(np.tanh(v))


@register("bpso")
def run(
    fitness: FitnessFn, dim: int, N: int, T: int, rng: np.random.Generator,
    c1: float = 2.0, c2: float = 2.0, wmax: float = 0.9, wmin: float = 0.4,
    vmax: float = 6.0,
) -> Result:
    """BPSO estándar con inercia lineal decreciente."""
    X = random_binary_population(N, dim, rng)
    V = np.zeros((N, dim))

    fit = np.array([fitness(x) for x in X])
    pbest, fit_p = X.copy(), fit.copy()
    g = int(np.argmin(fit_p))
    gbest, fit_g = pbest[g].copy(), float(fit_p[g])

    curve = np.empty(T)
    for t in range(1, T + 1):
        w = wmax - (wmax - wmin) * (t / T)
        for i in range(N):
            X[i], V[i] = update_particle(X[i], V[i], pbest[i], gbest, w, c1, c2, vmax, rng)
            f = fitness(X[i])
            if f < fit_p[i]:
                pbest[i], fit_p[i] = X[i].copy(), f
                if f < fit_g:
                    gbest, fit_g = X[i].copy(), f
        curve[t - 1] = fit_g

    return Result(mask=gbest.astype(bool), best_fitness=fit_g, curve=curve)
