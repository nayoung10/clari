# /// script
# requires-python = "==3.11.*"
# dependencies = ["csd-python-api>=3.7.0", "polars"]
#
# [[tool.uv.index]]
# name = "ccdc"
# url = "https://pip.ccdc.cam.ac.uk/"
#
# [tool.uv.sources]
# csd-python-api = { index = "ccdc" }
# ///
"""Symmetric GT CIF cache for COMPACK (packbench addition).

build_test_cifs_cache.py writes each GT crystal as CLARI's processed P1 unit cell. COMPACK is
several times slower against P1 cells than against the same structure with its space group. This
writes the same rows (csd_id, family, subsets) with the CSD entry's own CIF, which keeps the space
group. Use it with compack.py via CLARI_GT_CIFS=<out>.
"""

import os
from pathlib import Path

import polars as pl
from ccdc.io import EntryReader

DATA_DIR = Path(os.environ.get("CLARI_DATA_DIR", Path.cwd() / "data"))


def main() -> None:
    src = DATA_DIR / "csd" / "test_cifs.parquet"
    out = DATA_DIR / "csd" / "test_cifs_csd.parquet"
    df = pl.read_parquet(src)
    reader = EntryReader("CSD")
    cifs = [reader.entry(cid).crystal.to_string("cif") for cid in df["csd_id"]]
    df.with_columns(pl.Series("cif", cifs)).write_parquet(out)
    print(f"Wrote {len(df)} symmetric GT CIFs to {out}")


if __name__ == "__main__":
    main()
