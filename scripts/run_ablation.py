"""Etapa 3: sensibilidad al número de especies (ns) y ablación de CBPSO-MIWS.

Uso (desde la raíz del repo):
    uv run python scripts/run_ablation.py              # 30 corridas (~10 min en paralelo)
    uv run python scripts/run_ablation.py --runs 2     # prueba rápida

Se corre en las 3 bases con margen de mejora (34-90 variables); en las de
9 variables todos los métodos llegan casi siempre al mismo óptimo.

Incluye también ``bpso`` y ``cbpso_miws`` para que TODAS las comparaciones de
la etapa 3 salgan de la misma máquina (pareo exacto por corrida; ver
``bfs.exact.rescore`` sobre los empates de KNN entre computadoras).

Salida: ``results/etapa3/`` con ``runs.csv``, ``curves.npz`` y ``evals.npz``
(mismo formato que ``results/``). Es reanudable.
"""

from __future__ import annotations

import argparse

from bfs.experiment import ETAPA3_METHODS, run_experiment

DATASETS = ("ionosphere", "libras", "lung_cancer")
METHODS = ("bpso", "cbpso_miws", *ETAPA3_METHODS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default="results/etapa3")
    args = parser.parse_args()

    df = run_experiment(datasets=DATASETS, methods=METHODS, runs=args.runs,
                        out_dir=args.out, workers=args.workers)
    print(df.groupby(["dataset", "method"])[["best_fitness", "accuracy", "n_features"]]
            .mean().round(4).unstack("dataset"))


if __name__ == "__main__":
    main()
