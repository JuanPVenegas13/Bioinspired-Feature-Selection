"""Corre los experimentos y guarda results/runs.csv y results/curves.npz.

Uso (desde la raíz del repo):

    # prueba rápida (2 corridas, una base, dos métodos)
    uv run python scripts/run_experiments.py --runs 2 --datasets ionosphere \
        --methods bpso cbpso_miws

    # todo: 7 bases x 6 configuraciones x 30 corridas = 1,260 corridas
    uv run python scripts/run_experiments.py

Es **reanudable**: si results/runs.csv ya existe, solo corre las
combinaciones (base, método, corrida) que falten. Para empezar de cero,
borra results/runs.csv y results/curves.npz (o usa otro --out).

La lógica vive en src/bfs/experiment.py; aquí solo está la línea de comandos.
"""

from __future__ import annotations

import argparse
import time

from bfs.data import DATASETS
from bfs.experiment import METHODS, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimentos de la réplica de CBPSO-MIWS (Too et al., 2019).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs", type=int, default=30,
                        help="corridas independientes por configuración (la materia pide >= 30)")
    parser.add_argument("--datasets", nargs="+", default=None, choices=sorted(DATASETS),
                        metavar="BASE", help=f"bases a correr (todas: {', '.join(sorted(DATASETS))})")
    parser.add_argument("--methods", nargs="+", default=None, choices=sorted(METHODS),
                        metavar="METODO", help=f"configuraciones (todas: {', '.join(sorted(METHODS))})")
    parser.add_argument("--N", type=int, default=10, help="tamaño de población (Tabla 2)")
    parser.add_argument("--T", type=int, default=100, help="iteraciones (Tabla 2)")
    parser.add_argument("--out", default="results", help="carpeta de salida")
    parser.add_argument("--workers", type=int, default=None,
                        help="procesos en paralelo (por defecto, todos los núcleos; 1 = secuencial)")
    parser.add_argument("--save-every", type=int, default=25,
                        help="cada cuántas tareas se guarda el progreso")
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = run_experiment(
        datasets=args.datasets, methods=args.methods, runs=args.runs,
        N=args.N, T=args.T, out_dir=args.out, workers=args.workers,
        save_every=args.save_every,
    )
    minutos = (time.perf_counter() - t0) / 60

    print(f"\nListo en {minutos:.1f} min · {len(df)} filas en {args.out}/runs.csv")
    resumen = (
        df.groupby(["dataset", "method"])[["accuracy", "best_fitness", "n_evals"]]
        .mean()
        .round(4)
    )
    print("\nPromedios por base y método:")
    print(resumen.to_string())


if __name__ == "__main__":
    main()
