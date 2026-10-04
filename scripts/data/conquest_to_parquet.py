# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "polars",
#   "jsonargparse",
# ]
# ///
"""Pack ConQuest multi-entry .cif/.mol2 exports into raw/csd_conquest.parquet (id, cif, mol2).

Not part of upstream: the README says to export all of CSD with ConQuest into
csd_conquest.parquet but does not include this step.
"""

import pathlib
import re

import polars as pl
from jsonargparse import auto_cli


def split_cif(text):
    # Each entry starts with "data_CSD_CIF_<REFCODE>"
    blocks = {}
    for match in re.finditer(r"^data_CSD_CIF_(\S+)\n.*?(?=^data_|\Z)", text, flags=re.M | re.S):
        blocks[match.group(1)] = match.group(0)
    return blocks


def split_mol2(text):
    # Each entry starts with "@<TRIPOS>MOLECULE", followed by the refcode on the next line
    blocks = {}
    for match in re.finditer(r"^@<TRIPOS>MOLECULE\n(\S+).*?(?=^@<TRIPOS>MOLECULE|\Z)", text, flags=re.M | re.S):
        blocks[match.group(1)] = match.group(0)
    return blocks


def main(exports: list[str], out: str = str(pathlib.Path("data/raw/csd_conquest.parquet"))):
    """exports: ConQuest job prefixes, e.g. path/to/job for job.cif + job.mol2."""
    cifs, mol2s = {}, {}
    for prefix in exports:
        cifs.update(split_cif(pathlib.Path(f"{prefix}.cif").read_text()))
        mol2s.update(split_mol2(pathlib.Path(f"{prefix}.mol2").read_text()))

    ids = sorted(cifs.keys() & mol2s.keys())
    print(f"cif: {len(cifs)}, mol2: {len(mol2s)}, both: {len(ids)}")
    for name, only in [("cif only", cifs.keys() - mol2s.keys()), ("mol2 only", mol2s.keys() - cifs.keys())]:
        if only:
            print(f"{name}: {len(only)} (e.g. {sorted(only)[:5]})")

    df = pl.DataFrame({"id": ids, "cif": [cifs[i] for i in ids], "mol2": [mol2s[i] for i in ids]})
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"Wrote {len(df)} entries to {out}")


if __name__ == "__main__":
    auto_cli(main)
