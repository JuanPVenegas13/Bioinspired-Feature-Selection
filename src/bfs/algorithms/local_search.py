"""Mejora 2: búsqueda local sobre el Gbest de CBPSO-MIWS (algoritmo memético).

Motivación. El artículo (Too et al., 2019) usa la función de transferencia
sigmoide (ec. 2-3), que con ``v = 0`` deja cada bit al azar: BPSO ya es muy
exploratorio, y la co-evolución de la sección 3 agrega todavía más diversidad.
Las pruebas de la mejora 1 (función en V) confirmaron que **agregar
exploración no ayuda**. Esta mejora va en la dirección contraria: después de
cada iteración se intenta refinar la mejor solución global volteando bits uno
a uno (*bit-flip hill climbing* con criterio de **primera mejora**). El PSO
explora el espacio y la búsqueda local explota los vecindarios prometedores.

Costo. Cada iteración gasta como máximo ``n_flips`` evaluaciones extra de
fitness. Con los valores de la Tabla 2 (N = 10, T = 100) la réplica gasta
~1,010 evaluaciones y esta variante ~1,500; por eso el runner incluye la
configuración de control ``cbpso_miws_T150``, que iguala el presupuesto.
"""

from __future__ import annotations

import numpy as np

from bfs.algorithms.base import FitnessFn, Result, register
from bfs.algorithms.cbpso_miws import run as run_cbpso_miws


def bit_flip_local_search(
    mask: np.ndarray, fit: float, fitness: FitnessFn,
    rng: np.random.Generator, n_flips: int = 5,
) -> tuple[np.ndarray, float]:
    """Hill climbing de un bit con criterio de primera mejora.

    Prueba hasta ``n_flips`` bits distintos de ``mask``, en orden aleatorio
    (``rng.permutation``). En cuanto un volteo reduce el fitness lo acepta y
    regresa de inmediato (primera mejora, no la mejor mejora).

    Parámetros
    ----------
    mask : máscara binaria de la solución a refinar (no se modifica).
    fit : fitness de ``mask``, ya conocido (así no se vuelve a evaluar).
    fitness : función objetivo a MINIMIZAR (ver ``bfs.fitness``).
    rng : generador del que sale TODA la aleatoriedad.
    n_flips : bits a probar como máximo; se acota a ``dim``.

    Regresa
    -------
    ``(mejor_mask, mejor_fit)``. Nunca es peor que la entrada: si ningún
    volteo mejora, regresa la máscara y el fitness originales.

    Gasta como máximo ``n_flips`` evaluaciones de fitness y consume siempre
    la misma cantidad de ``rng`` (una permutación), así que es reproducible.
    """
    dim = int(mask.size)
    n_flips = min(int(n_flips), dim)
    orden = rng.permutation(dim)[:n_flips]

    for j in orden:
        vecino = mask.copy()
        vecino[j] = 1 - vecino[j]       # voltea el bit j
        f = float(fitness(vecino))
        if f < fit:
            return vecino, f            # primera mejora: acepta y regresa

    return mask, fit                    # ningún vecino mejora


class _FitnessContado:
    """Envuelve la función de fitness para contar cuántas veces se llamó."""

    def __init__(self, fn: FitnessFn):
        self.fn = fn
        self.n = 0

    def __call__(self, mask: np.ndarray) -> float:
        self.n += 1
        return self.fn(mask)


class BitFlipHook:
    """``gbest_hook`` que además lleva la cuenta de lo que gasta.

    ``cbpso_miws.run`` llama al hook **exactamente una vez por iteración**, así
    que ``evals[t]`` son las evaluaciones que la búsqueda local gastó en la
    iteración ``t``. Con eso se reconstruye el eje de **evaluaciones** de la
    curva de convergencia, necesario para comparar de forma justa contra
    configuraciones con otro número de iteraciones (``cbpso_miws_T150``).

    Es equivalente a ``partial(bit_flip_local_search, n_flips=...)``: no cambia
    el resultado ni el consumo de ``rng``, solo observa.
    """

    def __init__(self, n_flips: int = 5):
        self.n_flips = n_flips
        self.evals: list[int] = []

    def __call__(self, mask, fit, fitness, rng) -> tuple[np.ndarray, float]:
        contado = _FitnessContado(fitness)
        nueva, nuevo_fit = bit_flip_local_search(mask, fit, contado, rng, self.n_flips)
        self.evals.append(contado.n)
        return nueva, nuevo_fit


def _run_con_busqueda_local(
    fitness: FitnessFn, dim: int, N: int, T: int, rng: np.random.Generator,
    n_flips: int, transfer: str, **params,
) -> Result:
    """CBPSO-MIWS + búsqueda local, anotando el costo por iteración."""
    hook = BitFlipHook(n_flips=n_flips)
    res = run_cbpso_miws(fitness, dim, N, T, rng, **params,
                         gbest_hook=hook, transfer=transfer)
    res.extra["ls_evals"] = np.asarray(hook.evals, dtype=int)
    res.extra["n_flips"] = int(min(n_flips, dim))
    return res


@register("cbpso_miws_ls")
def run_ls(
    fitness: FitnessFn, dim: int, N: int, T: int, rng: np.random.Generator,
    n_flips: int = 5, **params,
) -> Result:
    """Mejora 2: réplica (sigmoide) + búsqueda local sobre el Gbest.

    Todo lo demás (especies, inercia múltiple, Tabla 2) es idéntico a
    ``cbpso_miws``; el único añadido es el ``gbest_hook``.

    ``extra["ls_evals"]`` trae las evaluaciones que gastó la búsqueda local en
    cada iteración (ver ``BitFlipHook``).
    """
    return _run_con_busqueda_local(fitness, dim, N, T, rng, n_flips, "s", **params)


@register("cbpso_miws_v_ls")
def run_v_ls(
    fitness: FitnessFn, dim: int, N: int, T: int, rng: np.random.Generator,
    n_flips: int = 5, **params,
) -> Result:
    """Mejoras 1 + 2: función de transferencia en V + búsqueda local.

    Sirve para ver si se complementan: la V explora de forma más conservadora
    (ver ``bpso.v_shaped``) y la búsqueda local explota.
    """
    return _run_con_busqueda_local(fitness, dim, N, T, rng, n_flips, "v", **params)
