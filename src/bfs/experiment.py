"""Runner de experimentos: 30 corridas independientes por configuración.

Requisito de la materia: ≥ 30 corridas por configuración (el artículo usó 20),
reportando media, mediana y desviación estándar, y comparando con Wilcoxon.

Pareo de semillas (necesario para el Wilcoxon **pareado** de ``bfs.stats``):
la corrida ``r`` usa la **misma** partición 80/20 (``holdout_split(ds, seed=r)``)
y el **mismo** generador (``np.random.default_rng(1000 + r)``) en **todos** los
métodos. Así las diferencias entre métodos no se deben a la partición.

Cada tarea ``(dataset, method, run)`` es independiente: crea su propio
``FeatureSelectionProblem`` (para que ``n_evals`` empiece en 0) y su propio
``rng``. Por eso el resultado **no depende** del orden ni del número de
procesos: paralelizar no cambia los números.

Salidas (contrato con la etapa 3, ver FASE2 sección 3):
  * ``results/runs.csv``   — una fila por (dataset, método, corrida)
  * ``results/curves.npz`` — clave ``f"{dataset}__{method}"`` -> (n_runs, T)
    con la curva de convergencia (fitness) de cada corrida

Además, fuera del contrato (archivo adicional, por si Arie lo quiere usar):

  * ``results/evals.npz`` — mismas claves, pero con las **evaluaciones de
    fitness acumuladas** al terminar cada iteración. Es lo que permite
    graficar la convergencia contra evaluaciones y comparar de forma justa
    configuraciones con distinto presupuesto.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from bfs.algorithms import ALGORITHMS
from bfs.data import DATASETS, Dataset, holdout_split, load_dataset
from bfs.fitness import FeatureSelectionProblem

# --------------------------------------------------------------------------
# Configuraciones
# --------------------------------------------------------------------------

METHODS: dict[str, tuple[str, dict]] = {
    # nombre en runs.csv : (algoritmo en ALGORITHMS, parámetros)
    "bpso":             ("bpso", {}),                # línea base (sección 2)
    "cbpso_miws":       ("cbpso_miws", {}),          # la réplica (Algoritmo 1)
    "cbpso_miws_v":     ("cbpso_miws_v", {}),        # mejora 1: función en V
    "cbpso_miws_ls":    ("cbpso_miws_ls", {}),       # mejora 2: búsqueda local
    "cbpso_miws_v_ls":  ("cbpso_miws_v_ls", {}),     # mejoras 1 + 2
    "cbpso_miws_T150":  ("cbpso_miws", {"T": 150}),  # control de presupuesto
}
"""Configuraciones a correr. Agregar una fila aquí es todo lo necesario: el
runner, el CSV y la estadística la recogen sin más cambios. Ejemplos para la
etapa 3 (sensibilidad y ablación)::

    "cbpso_ns1":  ("cbpso_miws", {"ns": 1}),          # sin co-evolución
    "cbpso_iws1": ("cbpso_miws", {"schemes": (1,)}),  # sin inercia múltiple

Las claves ``N`` y ``T`` del diccionario de parámetros sobrescriben los
valores generales (así funciona el control ``cbpso_miws_T150``).
"""

ETAPA3_METHODS: dict[str, tuple[str, dict]] = {
    # Sensibilidad al número de especies (el artículo dice que ns = 3 y 4 fueron
    # los mejores, pero no muestra los resultados). ns = 3 es "cbpso_miws".
    "cbpso_ns1":  ("cbpso_miws", {"ns": 1}),   # 1 especie de 10 = sin co-evolución
    "cbpso_ns2":  ("cbpso_miws", {"ns": 2}),   # 5 + 5
    "cbpso_ns4":  ("cbpso_miws", {"ns": 4}),   # 3 + 3 + 2 + 2
    "cbpso_ns5":  ("cbpso_miws", {"ns": 5}),   # 2 + 2 + 2 + 2 + 2
    # Ablación de la inercia múltiple: co-evolución (ns = 3) con UN solo esquema.
    "cbpso_iws1": ("cbpso_miws", {"schemes": (1,)}),   # lineal decreciente (ec. 6)
    "cbpso_iws2": ("cbpso_miws", {"schemes": (2,)}),   # aleatoria (ec. 7)
    "cbpso_iws3": ("cbpso_miws", {"schemes": (3,)}),   # no lineal (ec. 8)
    "cbpso_iws4": ("cbpso_miws", {"schemes": (4,)}),   # constante (ec. 9)
}
"""Configuraciones de la etapa 3 (Arie). Se corren aparte, en ``results/etapa3/``
(``scripts/run_ablation.py``); no forman parte de la corrida por defecto ni del
notebook 03. Ablación completa (2 × 2):

    co-evolución   inercia múltiple   configuración
    sí             sí                 cbpso_miws   (la réplica)
    no             sí                 cbpso_ns1
    sí             no                 cbpso_iws1
    no             no                 bpso         (BPSO estándar, misma inercia que IWS 1)
"""

ALL_METHODS: dict[str, tuple[str, dict]] = {**METHODS, **ETAPA3_METHODS}
"""Todo lo que el runner sabe correr. Sin ``methods``, se corre solo ``METHODS``."""

RUN_COLUMNS = [
    "dataset", "method", "run", "split_seed", "algo_seed",
    "best_fitness", "worst_fitness", "mean_fitness", "std_fitness",
    "accuracy", "n_features", "n_evals", "time_s", "selected",
]
"""Columnas de ``runs.csv``, en orden (contrato con la etapa 3)."""

ALGO_SEED_BASE = 1000
"""La corrida ``r`` usa el generador ``default_rng(ALGO_SEED_BASE + r)``."""

KEY_SEP = "__"
"""Separador de las claves de ``curves.npz``: ``f"{dataset}{KEY_SEP}{method}"``."""


def curve_key(dataset: str, method: str) -> str:
    """Clave de ``curves.npz`` para un par (base, método)."""
    return f"{dataset}{KEY_SEP}{method}"


# --------------------------------------------------------------------------
# Una corrida
# --------------------------------------------------------------------------

@cache
def _load_cached(dataset: str) -> Dataset:
    """``load_dataset`` con caché: dentro de un proceso la base se lee una vez."""
    return load_dataset(dataset)


def evals_curve(N: int, T: int, ls_evals: np.ndarray | None = None) -> np.ndarray:
    """Evaluaciones de fitness acumuladas al terminar cada iteración.

    El PSO gasta ``N`` evaluaciones al inicializar (paso 3) y ``N`` por
    iteración, así que al terminar la iteración ``t`` lleva ``N + N t``. La
    búsqueda local agrega lo que reporta ``BitFlipHook.evals``.

    Sirve para graficar la convergencia contra **evaluaciones** en vez de
    iteraciones: es la única comparación justa entre configuraciones con
    distinto presupuesto (``cbpso_miws_ls`` vs. ``cbpso_miws_T150``).
    """
    acumuladas = N + N * np.arange(1, T + 1, dtype=np.int64)
    if ls_evals is not None:
        acumuladas = acumuladas + np.cumsum(np.asarray(ls_evals, dtype=np.int64))
    return acumuladas


def run_one(dataset: str, method: str, run: int, N: int = 10, T: int = 100,
            ) -> tuple[dict, np.ndarray, np.ndarray]:
    """Una corrida -> (fila de ``runs.csv``, curva de fitness, curva de evaluaciones).

    Las ecuaciones 13-16 del artículo se calculan **sobre la curva de esta
    corrida** (así las define el artículo); promediarlas entre corridas es
    trabajo de ``bfs.stats``.
    """
    if dataset not in DATASETS:
        raise KeyError(f"Base desconocida {dataset!r}. Opciones: {sorted(DATASETS)}")
    if method not in ALL_METHODS:
        raise KeyError(f"Método desconocido {method!r}. Opciones: {sorted(ALL_METHODS)}")

    algo_name, params = ALL_METHODS[method]
    params = dict(params)                      # copia: no mutar METHODS
    N_eff = int(params.pop("N", N))
    T_eff = int(params.pop("T", T))

    ds = _load_cached(dataset)
    split_seed, algo_seed = run, ALGO_SEED_BASE + run
    problem = FeatureSelectionProblem(holdout_split(ds, seed=split_seed))
    rng = np.random.default_rng(algo_seed)

    res = ALGORITHMS[algo_name](problem, problem.dim, N_eff, T_eff, rng, **params)

    curve = np.asarray(res.curve, dtype=float)
    row = {
        "dataset": dataset,
        "method": method,
        "run": int(run),
        "split_seed": int(split_seed),
        "algo_seed": int(algo_seed),
        "best_fitness": float(curve.min()),          # ec. 13
        "worst_fitness": float(curve.max()),         # ec. 14
        "mean_fitness": float(curve.mean()),         # ec. 15
        "std_fitness": float(curve.std(ddof=0)),     # ec. 16
        "accuracy": float(problem.accuracy(res.mask)),   # ec. 17, en %
        "n_features": int(res.n_selected),
        "n_evals": int(problem.n_evals),
        "time_s": float(res.elapsed),
        "selected": ";".join(str(i) for i in res.selected),
    }

    evals = evals_curve(N_eff, T_eff, res.extra.get("ls_evals"))
    # Cuadratura del costo: el eje de evaluaciones debe cerrar con el contador
    # real del problema. Si no cierra, el modelo de costo está mal.
    if int(evals[-1]) != problem.n_evals:
        raise AssertionError(
            f"{dataset}/{method}/run {run}: la curva de evaluaciones termina en "
            f"{int(evals[-1])} pero problem.n_evals = {problem.n_evals}"
        )
    return row, curve, evals


def _task(args: tuple[str, str, int, int, int]) -> tuple[dict, np.ndarray, np.ndarray]:
    """Envoltura picklable para ``ProcessPoolExecutor``."""
    dataset, method, run, N, T = args
    return run_one(dataset, method, run, N=N, T=T)


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------

def _apilar(por_par: dict[str, dict[int, np.ndarray]]) -> dict[str, np.ndarray]:
    """``{clave: {run: curva}}`` -> ``{clave: (n_runs, T)}``, fila r = corrida r."""
    return {
        key: np.vstack([por_run[r] for r in sorted(por_run)])
        for key, por_run in por_par.items() if por_run
    }


def _save(df: pd.DataFrame, curves: dict[str, dict[int, np.ndarray]],
          evals: dict[str, dict[int, np.ndarray]], out_dir: Path) -> pd.DataFrame:
    """Guarda ``runs.csv`` (ordenado), ``curves.npz`` y ``evals.npz``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["dataset", "method", "run"]).reset_index(drop=True)
    df = df[RUN_COLUMNS]
    df.to_csv(out_dir / "runs.csv", index=False)

    # Una entrada por par: fila r = curva de la corrida r (las corridas van 0..n-1)
    np.savez_compressed(out_dir / "curves.npz", **_apilar(curves))
    # Archivo ADICIONAL (no es parte del contrato de la sección 3): mismas
    # claves, pero con las evaluaciones acumuladas de cada iteración.
    np.savez_compressed(out_dir / "evals.npz", **_apilar(evals))
    return df


def _load_previous(
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[int, np.ndarray]], dict[str, dict[int, np.ndarray]]]:
    """Lee ``runs.csv``, ``curves.npz`` y ``evals.npz`` previos para reanudar."""
    csv_path = out_dir / "runs.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=RUN_COLUMNS), {}, {}

    df = pd.read_csv(csv_path)
    faltan = set(RUN_COLUMNS) - set(df.columns)
    if faltan:
        raise ValueError(
            f"{csv_path} no sigue el contrato (faltan columnas: {sorted(faltan)}). "
            "Bórralo o usa otro --out."
        )
    df["selected"] = df["selected"].fillna("")      # máscara vacía -> cadena vacía

    def leer(nombre: str) -> dict[str, dict[int, np.ndarray]]:
        ruta = out_dir / nombre
        if not ruta.exists():
            return {}
        por_par: dict[str, dict[int, np.ndarray]] = {}
        with np.load(ruta) as z:
            for key in z.files:
                arreglo = z[key]
                # Se reconstruye run -> curva con las corridas que hay en el CSV.
                dataset, _, method = key.partition(KEY_SEP)
                hechas = sorted(
                    df.loc[(df["dataset"] == dataset) & (df["method"] == method), "run"]
                )
                if len(hechas) == len(arreglo):
                    por_par[key] = {int(r): arreglo[i] for i, r in enumerate(hechas)}
        return por_par

    return df, leer("curves.npz"), leer("evals.npz")


# --------------------------------------------------------------------------
# Experimento completo
# --------------------------------------------------------------------------

def run_experiment(
    datasets: list[str] | tuple[str, ...] | None = None,
    methods: list[str] | tuple[str, ...] | None = None,
    runs: int = 30,
    N: int = 10,
    T: int = 100,
    out_dir: str | Path = "results",
    workers: int | None = None,
    save_every: int = 25,
    verbose: bool = True,
) -> pd.DataFrame:
    """Corre todo, guarda ``runs.csv``, ``curves.npz`` y ``evals.npz``.

    Parámetros
    ----------
    datasets, methods : claves de ``DATASETS`` / ``METHODS`` (por defecto, todas).
    runs : número de corridas independientes por configuración (30).
    N, T : población e iteraciones (Tabla 2); los parámetros de ``METHODS``
        pueden sobrescribirlos (``cbpso_miws_T150``).
    out_dir : carpeta de salida.
    workers : procesos en paralelo. ``None`` = todos los núcleos; ``1`` =
        secuencial (sin ``ProcessPoolExecutor``, útil para depurar).
    save_every : cada cuántas tareas terminadas se guarda el progreso.
    verbose : imprimir el avance.

    **Reanudable**: las combinaciones ``(dataset, method, run)`` que ya estén
    en ``runs.csv`` no se repiten.
    """
    datasets = list(DATASETS) if datasets is None else list(datasets)
    methods = list(METHODS) if methods is None else list(methods)
    out_dir = Path(out_dir)

    for d in datasets:
        if d not in DATASETS:
            raise KeyError(f"Base desconocida {d!r}. Opciones: {sorted(DATASETS)}")
    for m in methods:
        if m not in ALL_METHODS:
            raise KeyError(f"Método desconocido {m!r}. Opciones: {sorted(ALL_METHODS)}")

    df, curves, evals = _load_previous(out_dir)
    ya_hechas = set(zip(df["dataset"], df["method"], df["run"])) if len(df) else set()

    tareas = [
        (d, m, r, N, T)
        for d in datasets for m in methods for r in range(runs)
        if (d, m, r) not in ya_hechas
    ]
    total = len(tareas)
    if verbose:
        n_pedidas = len(datasets) * len(methods) * runs
        print(
            f"{n_pedidas} corridas pedidas · {n_pedidas - total} ya estaban · "
            f"{total} por correr"
        )
    if total == 0:
        return _save(df, curves, evals, out_dir)

    filas: list[dict] = []

    def registrar(fila: dict, curva: np.ndarray, curva_evals: np.ndarray, i: int) -> None:
        filas.append(fila)
        key = curve_key(fila["dataset"], fila["method"])
        curves.setdefault(key, {})[fila["run"]] = curva
        evals.setdefault(key, {})[fila["run"]] = curva_evals
        if verbose:
            print(
                f"[{i}/{total}] {fila['dataset']} {fila['method']} "
                f"run {fila['run']} — {fila['time_s']:.1f} s",
                flush=True,
            )

    def volcar() -> pd.DataFrame:
        nonlocal df, filas
        if filas:
            nuevas = pd.DataFrame(filas, columns=RUN_COLUMNS)
            df = nuevas if df.empty else pd.concat([df, nuevas], ignore_index=True)
            filas = []
        return _save(df, curves, evals, out_dir)

    if workers == 1:
        for i, args in enumerate(tareas, start=1):
            fila, curva, curva_evals = _task(args)
            registrar(fila, curva, curva_evals, i)
            if i % save_every == 0:
                volcar()
    else:
        # Un hilo de BLAS por proceso: con un proceso por núcleo, varios hilos
        # por proceso se pelean y hacen todo más lento. Los hijos heredan el
        # entorno, así que esto debe quedar fijado ANTES de crear el pool.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(var, "1")

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futuros = [pool.submit(_task, args) for args in tareas]
            for i, fut in enumerate(as_completed(futuros), start=1):
                fila, curva, curva_evals = fut.result()
                registrar(fila, curva, curva_evals, i)
                if i % save_every == 0:
                    volcar()

    return volcar()
