# Bioinspired Feature Selection — Réplica de CBPSO-MIWS

Réplica en Python del artículo:

> Too, J., Abdullah, A. R., & Mohd Saad, N. (2019). *A New Co-Evolution Binary Particle Swarm Optimization with Multiple Inertia Weight Strategy for Feature Selection*. Informatics, 6(2), 21. https://doi.org/10.3390/informatics6020021

Proyecto de MA2015 Design of Bioinspired Mathematical Algorithms — Tec de Monterrey, Campus Guadalajara.

**Equipo:** Arie Zalo González Álvarez · Juan Pablo Venegas

## Instalación

Requiere [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/JuanPVenegas13/Bioinspired-Feature-Selection.git
cd Bioinspired-Feature-Selection
uv sync                                   # crea .venv con las versiones exactas de uv.lock
uv run python scripts/download_data.py    # descarga las bases de UCI a data/raw/
uv run pytest                             # verifica que todo funcione
```

## Bases de datos

Subconjunto de 6 de las 10 bases de la Tabla 1 del artículo (UCI Machine Learning Repository):

| # | Base | Instancias | Variables | Clases |
|---|---|---|---|---|
| 1 | Breast Cancer Wisconsin (Original) | 699 | 9 | 2 |
| 3 | Glass Identification | 214 | 9 (+ID) | 6 |
| 4 | Ionosphere | 351 | 34 | 2 |
| 5 | Libras Movement | 360 | 90 | 15 |
| 7 | Breast Cancer Coimbra | 116 | 9 | 2 |
| 8 | Lung Cancer | 32 | 56 | 3 |

## Estructura

```
src/bfs/          paquete con datos, fitness y algoritmos
  data.py           carga y preprocesamiento de las bases de UCI
  fitness.py        función de fitness del wrapper con KNN (ecs. 11-12)
  algorithms/       bpso, cbpso_miws y local_search (mejora 2)
  experiment.py     runner de las 30 corridas (paralelo y reanudable)
  stats.py          resumen descriptivo y Wilcoxon pareado
scripts/          descarga de datos y ejecución de experimentos
notebooks/        validación y análisis de resultados
results/          resultados de los experimentos (CSV, NPZ y figuras)
tests/            pruebas con pytest
```

## Métodos

| Nombre | Descripción | Autor |
|---|---|---|
| `bpso` | BPSO estándar (sección 2 del artículo), línea base | Arie |
| `cbpso_miws` | CBPSO-MIWS (Algoritmo 1), la réplica | Arie |
| `cbpso_miws_v` | Mejora 1: función de transferencia en V | Arie |
| `cbpso_miws_ls` | Mejora 2: búsqueda local sobre el Gbest | Juan Pablo |
| `cbpso_miws_v_ls` | Mejoras 1 + 2 combinadas | Juan Pablo |
| `cbpso_miws_T150` | Control de presupuesto (`T = 150`, iguala las evaluaciones) | Juan Pablo |

## Experimentos

```bash
# prueba rápida (4 corridas, ~5 s)
uv run python scripts/run_experiments.py --runs 2 --datasets ionosphere --methods bpso cbpso_miws

# todo: 7 bases × 6 configuraciones × 30 corridas = 1,260 corridas (~1.5 min en 10 núcleos)
uv run python scripts/run_experiments.py
```

Flags: `--runs` (30), `--datasets`, `--methods`, `--N` (10), `--T` (100), `--out`
(`results`), `--workers` (todos los núcleos; `1` = secuencial), `--save-every` (25).

Es **reanudable**: las combinaciones `(base, método, corrida)` que ya estén en
`results/runs.csv` no se repiten. Para empezar de cero, borra los archivos de
`results/`. Cada corrida crea su propio problema y su propio generador, así que
**el resultado no depende del número de procesos**.

La corrida *r* usa la partición `seed=r` y el generador `1000 + r` en **todas** las
configuraciones: eso permite comparar con Wilcoxon **pareado**.

### Salidas

| Archivo | Qué contiene |
|---|---|
| `results/runs.csv` | una fila por (base, método, corrida): ecs. 13-16, exactitud, nº de variables, evaluaciones, tiempo y variables seleccionadas |
| `results/curves.npz` | clave `"{base}__{método}"` → curvas de convergencia `(n_corridas, T)` |
| `results/evals.npz` | mismas claves → evaluaciones acumuladas por iteración (para graficar contra presupuesto) |
| `results/figures/*.png` | figuras del póster a 300 dpi (las genera el notebook 03) |
| `results/resumen.csv`, `results/wilcoxon.csv` | tablas del notebook 03 |

### Análisis

`notebooks/03_mejoras.ipynb` arma las tablas, los boxplots y las curvas de convergencia
a partir de `results/` (no vuelve a correr experimentos) y guarda las figuras del póster
en `results/figures/`. Está guardado **con salidas**, así que se puede leer sin ejecutar
nada; para volver a correrlo, ábrelo en VS Code con el kernel `.venv` del proyecto.

`src/bfs/stats.py` tiene `summarize` (media, mediana, desv.), `wilcoxon_paired`
(Wilcoxon de rangos con signo pareado por corrida) y `holm` (corrección por
comparaciones múltiples).

### Agregar una configuración

Basta con una línea en `METHODS` (`src/bfs/experiment.py`); el runner, el CSV y la
estadística la recogen sin más cambios:

```python
"cbpso_ns1": ("cbpso_miws", {"ns": 1}),   # sin co-evolución
```

## Estado

- [x] Etapa 1 (Arie) — datos, fitness, BPSO, CBPSO-MIWS, notebook 01, mejora 1 (función en V)
- [x] Etapa 2 (Juan Pablo) — mejora 2 (búsqueda local), runner de 30 corridas, Wilcoxon, notebook 03
- [x] Etapa 3 (Arie) — réplica vs. artículo, óptimo exacto, sensibilidad a `ns`, ablación, notebook 02
- [ ] Etapa 4 (ambos) — póster

## Problemas comunes

**`ModuleNotFoundError: No module named 'bfs'` (macOS)**

Pasa cuando el repo está en una carpeta sincronizada con iCloud (`Desktop`, `Documents`): macOS marca como oculto el archivo `.pth` que uv usa para instalar el paquete, y Python lo ignora. Solución:

```bash
chflags -R nohidden .venv
```

Si vuelve a pasar después de un `uv sync`, repite el comando. Para evitarlo por completo, clona el repo fuera de iCloud (por ejemplo en `~/Proyectos/`).