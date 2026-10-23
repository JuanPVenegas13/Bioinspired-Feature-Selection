"""Calcula el óptimo exacto (enumeración) de las bases con 9-10 variables.

Uso (desde la raíz del repo):
    uv run python scripts/exact_optimum.py

Salidas:
    results/optimo_exacto.csv  — óptimo de cada (base, corrida)
    results/tasa_acierto.csv   — % de corridas en que cada método llegó al óptimo
                                 (requiere results/runs.csv del runner)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bfs.exact import compute_optima, hit_rate

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    optima = compute_optima()
    optima.to_csv(RESULTS / "optimo_exacto.csv", index=False)
    print(f"Guardado {RESULTS / 'optimo_exacto.csv'}")

    runs_csv = RESULTS / "runs.csv"
    if runs_csv.exists():
        tasa = hit_rate(pd.read_csv(runs_csv), optima)
        tasa.to_csv(RESULTS / "tasa_acierto.csv", index=False)
        print(f"Guardado {RESULTS / 'tasa_acierto.csv'}\n")
        print(tasa.pivot(index="method", columns="dataset", values="hit_rate").round(1))


if __name__ == "__main__":
    main()
