"""Descarga los archivos originales de UCI usados en la réplica a data/raw/.

Uso (desde la raíz del repo):
    uv run python scripts/download_data.py

Los archivos se guardan tal cual vienen de UCI (sin modificar); toda la
limpieza ocurre en src/bfs/data.py. Si un archivo ya existe, no se vuelve
a descargar (usa --force para sobrescribir).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"

# nombre de archivo local -> URL original en UCI
FILES = {
    "breast-cancer-wisconsin.data": f"{BASE}/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
    "glass.data": f"{BASE}/glass/glass.data",
    "ionosphere.data": f"{BASE}/ionosphere/ionosphere.data",
    "movement_libras.data": f"{BASE}/libras/movement_libras.data",
    "dataR2.csv": f"{BASE}/00451/dataR2.csv",
    "lung-cancer.data": f"{BASE}/lung-cancer/lung-cancer.data",
}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="vuelve a descargar todo")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for fname, url in FILES.items():
        dest = RAW_DIR / fname
        if dest.exists() and not args.force:
            print(f"[ok]   {fname} (ya existe)")
            continue
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[desc] {fname}  <-  {url}")
        except Exception as exc:  # noqa: BLE001
            failed.append(fname)
            print(f"[ERROR] {fname}: {exc}", file=sys.stderr)

    if failed:
        print(
            "\nNo se pudieron descargar: " + ", ".join(failed)
            + "\nDescárgalos a mano desde la página del dataset en UCI y "
            f"guárdalos en {RAW_DIR} con esos mismos nombres.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
