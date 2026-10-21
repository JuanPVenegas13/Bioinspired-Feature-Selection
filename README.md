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
scripts/          descarga de datos y ejecución de experimentos
notebooks/        validación y análisis de resultados
results/          resultados de los experimentos (CSV)
tests/            pruebas con pytest
```

## Estado

- [ ] Fase 1 — datos, función de fitness, BPSO y CBPSO-MIWS
- [ ] Fase 2 — GA, CBGWO, experimentos y análisis

## Problemas comunes

**`ModuleNotFoundError: No module named 'bfs'` (macOS)**

Pasa cuando el repo está en una carpeta sincronizada con iCloud (`Desktop`, `Documents`): macOS marca como oculto el archivo `.pth` que uv usa para instalar el paquete, y Python lo ignora. Solución:

```bash
chflags -R nohidden .venv
```

Si vuelve a pasar después de un `uv sync`, repite el comando. Para evitarlo por completo, clona el repo fuera de iCloud (por ejemplo en `~/Proyectos/`).