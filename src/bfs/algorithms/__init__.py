"""Algoritmos de selección de variables.

Cada módulo que se importa aquí se registra solo en ``ALGORITHMS``
(ver ``base.register``). Al agregar un algoritmo nuevo, impórtalo abajo.
"""

from bfs.algorithms.base import ALGORITHMS, Result, register
from bfs.algorithms import bpso, cbpso_miws, local_search  # noqa: F401

__all__ = ["ALGORITHMS", "Result", "register"]
