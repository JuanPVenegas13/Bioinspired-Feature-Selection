"""Contrato común para todos los algoritmos de selección de variables.

Cada algoritmo vive en su propio módulo (bpso.py, cbpso_miws.py, ga.py, ...)
y expone una función ``run`` con ESTA firma:

    def run(fitness, dim, N, T, rng, **params) -> Result

    fitness : callable(mask) -> float   (se minimiza; ver bfs.fitness)
    dim     : número de variables (longitud de la máscara)
    N       : tamaño de población (10 en el artículo)
    T       : número máximo de iteraciones (100 en el artículo)
    rng     : np.random.Generator — TODA la aleatoriedad sale de aquí,
              para que cada corrida sea reproducible con su semilla
    params  : parámetros propios del algoritmo (c1, c2, ns, CR, MR, ...)

y se registra con el decorador ``@register("nombre")`` para que el runner
de experimentos lo encuentre por nombre.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

FitnessFn = Callable[[np.ndarray], float]


@dataclass
class Result:
    """Salida de una corrida de un algoritmo."""

    mask: np.ndarray          # mejor solución encontrada (bool, longitud dim)
    best_fitness: float       # fitness de esa solución
    curve: np.ndarray         # curve[t] = mejor fitness al terminar la iteración t (longitud T)
    elapsed: float = 0.0      # segundos
    extra: dict = field(default_factory=dict)  # info opcional (p. ej. historial de IWS)

    @property
    def selected(self) -> np.ndarray:
        """Índices de las variables seleccionadas."""
        return np.flatnonzero(self.mask)

    @property
    def n_selected(self) -> int:
        return int(self.mask.sum())


class Algorithm(Protocol):
    def __call__(
        self, fitness: FitnessFn, dim: int, N: int, T: int,
        rng: np.random.Generator, **params,
    ) -> Result: ...


ALGORITHMS: dict[str, Algorithm] = {}


def register(name: str):
    """Decorador: ``@register("bpso")`` agrega la función a ALGORITHMS."""

    def deco(fn: Algorithm) -> Algorithm:
        if name in ALGORITHMS:
            raise ValueError(f"Algoritmo {name!r} ya registrado")

        def timed(fitness, dim, N, T, rng, **params) -> Result:
            t0 = time.perf_counter()
            res = fn(fitness, dim, N, T, rng, **params)
            res.elapsed = time.perf_counter() - t0
            return res

        timed.__name__ = fn.__name__
        timed.__doc__ = fn.__doc__
        ALGORITHMS[name] = timed
        return timed

    return deco


# -- utilidades compartidas ----------------------------------------------------

def random_binary_population(N: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Población inicial: cada bit es 1 con probabilidad 0.5 (como jBPSO.m)."""
    return (rng.random((N, dim)) > 0.5).astype(np.int8)


def sigmoid(v: np.ndarray) -> np.ndarray:
    """Función de transferencia S (ecuación 2)."""
    return 1.0 / (1.0 + np.exp(-v))
