# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "pyarrow", "jsonargparse"]
# ///
"""Pack validated ConQuest chunks into a single id/cif/mol2 Parquet file."""

import hashlib
import json
import pathlib
import re
import sys

from jsonargparse import auto_cli


def split_blocks(text, pattern):
    blocks = {}
    for match in re.finditer(pattern, text, flags=re.M | re.S):
        refcode = match.group(1)
        if refcode in blocks:
            raise ValueError(f"Duplicate refcode: {refcode}")
        blocks[refcode] = match.group(0)
    return blocks


def split_cif(text):
    return split_blocks(text, r"^data_CSD_CIF_(\S+)\n.*?(?=^data_|\Z)")


def split_mol2(text):
    return split_blocks(text, r"^@<TRIPOS>MOLECULE\n(\S+).*?(?=^@<TRIPOS>MOLECULE|\Z)")


def main(
    exports: list[str] | None = None,
    out: str = "data/raw/csd_conquest.parquet",
    manifest: str | None = None,
):
    """Use a completed export manifest, or legacy explicit job prefixes (strict matching)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if bool(exports) == bool(manifest):
        raise ValueError("Supply either exports or manifest")
    chunks = []
    expected = None
    if manifest:
        path = pathlib.Path(manifest)
        data = json.loads(path.read_text())
        expected = data["requested"]
        accounted = 0
        for chunk in data["chunks"]:
            accounted += chunk["requested"]
            if chunk["status"] == "skipped":
                expected -= 1
            elif chunk["status"] == "complete":
                expected -= len(chunk["unmatched"])
                chunks.append((path.parent / chunk["directory"] / "export", chunk))
            else:
                raise ValueError("Manifest contains unfinished chunks")
        if accounted != data["requested"]:
            raise ValueError("Manifest does not account for every input")
    else:
        chunks = [(pathlib.Path(prefix), None) for prefix in exports]

    output = pathlib.Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".parquet.tmp")
    schema = pa.schema([(name, pa.string()) for name in ("id", "cif", "mol2")])
    seen = set()
    try:
        with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
            for prefix, chunk in chunks:
                if chunk:
                    for ext, checksum in chunk["hashes"].items():
                        with pathlib.Path(f"{prefix}.{ext}").open("rb") as stream:
                            actual = hashlib.file_digest(stream, "sha256").hexdigest()
                        if actual != checksum:
                            raise ValueError(f"Damaged export {prefix}.{ext}; resume the exporter")
                cifs = split_cif(pathlib.Path(f"{prefix}.cif").read_text())
                mol2s = split_mol2(pathlib.Path(f"{prefix}.mol2").read_text())
                if cifs.keys() != mol2s.keys():
                    raise ValueError(f"CIF/MOL2 identifiers differ in {prefix}")
                if chunk:
                    gcd = pathlib.Path(f"{prefix}.gcd").read_text().split()
                    if len(gcd) != len(set(gcd)) or set(gcd) != set(cifs):
                        raise ValueError(f"GCD identifiers differ in {prefix}")
                    if len(cifs) + len(chunk["unmatched"]) != chunk["requested"]:
                        raise ValueError(f"Unaccounted entries in {prefix}")
                if seen.intersection(cifs):
                    raise ValueError(f"Duplicate entries across chunks at {prefix}")
                ids = sorted(cifs)
                seen.update(ids)
                writer.write_table(
                    pa.Table.from_pydict(
                        {"id": ids, "cif": [cifs[i] for i in ids], "mol2": [mol2s[i] for i in ids]},
                        schema=schema,
                    )
                )
        if expected is not None and len(seen) != expected:
            raise ValueError(f"Expected {expected} entries, packed {len(seen)}")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Wrote {len(seen):,} entries to {output}")


if __name__ == "__main__":
    args = sys.argv[1:]
    # Preserve the original positional list-of-prefixes CLI.
    if args and not args[0].startswith("-"):
        args.insert(0, "--exports")
    auto_cli(main, args=args)
