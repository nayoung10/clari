"""Recovery tests use a fake executable; scientific export gets a separate real smoke test."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

import polars as pl
import pytest

DATA_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(DATA_SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "export_conquest", DATA_SCRIPTS / "export_conquest.py"
)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)

FAKE = """#!/usr/bin/env python3
import pathlib, sys, os, time
if '-version' in sys.argv:
    print('Fake ConQuest 1'); sys.exit(0)
ids = pathlib.Path(sys.argv[sys.argv.index('-restrict')+1]).read_text().split()
with pathlib.Path('attempts').open('a') as f: f.write('run\\n')
if 'SLOWWW' in ids and not pathlib.Path('slow-once').exists():
    pathlib.Path('slow-once').write_text(str(os.getpid()))
    time.sleep(60)
if 'BADBAD' in ids:
    print('Traceback (most recent call last):\\n  export_filters/export_cif.py _export_entry\\nKeyError: 706')
    sys.exit(1)
if 'INFRAF' in ids:
    print('No space left on device'); sys.exit(1)
if ids == ['ABSENT']:
    print('WARNING: ABSENT not found\\nERROR: No entries found in restriction file ids.gcd')
    sys.exit(1)
ids = [x for x in ids if x != 'ABSENT']
pathlib.Path('export.cif').write_text(''.join('data_CSD_CIF_'+x+"\\n_chemical_formula_sum 'C'\\n" for x in ids))
pathlib.Path('export.mol2').write_text(''.join('@<TRIPOS>MOLECULE\\n'+x+'\\n1 0\\n' for x in ids))
pathlib.Path('export.gcd').write_text('\\n'.join(ids))
"""


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":fake")
    exe = tmp_path / "cqbatch"
    exe.write_text(FAKE)
    exe.chmod(0o755)
    db = tmp_path / "testdb"
    db.with_suffix(".inf").write_text("test version")
    pathlib.Path(str(db) + "_CIP.sqlite").write_text("database")
    metadata = tmp_path / "metadata.parquet"
    pl.DataFrame({"id": ["AAAAAA", "ABSENT", "BADBAD", "CCCCCC"]}).write_parquet(metadata)
    args = dict(
        metadata=str(metadata),
        out=str(tmp_path / "export"),
        cqbatch=str(exe),
        databases=[str(db)],
        chunk_size=4,
    )
    return tmp_path, args


def test_skip_resume_changed_workers_and_corruption(setup):
    root, args = setup
    export.export_main(**args, workers=1)
    manifest_path = root / "export/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    leaves = manifest["chunks"]
    assert [c["id"] for c in leaves if c["status"] == "skipped"] == ["BADBAD"]
    assert sum(len(c.get("unmatched", [])) for c in leaves) == 1
    attempts = {p: p.read_text() for p in root.rglob("attempts")}
    export.export_main(**args, workers=4)
    assert attempts == {p: p.read_text() for p in root.rglob("attempts")}
    assert json.loads(manifest_path.read_text()) == manifest
    complete = next(c for c in leaves if c["status"] == "complete")
    damaged = root / "export" / complete["directory"] / "export.cif"
    original = damaged.read_text()
    damaged.write_text("truncated")
    export.export_main(**args, workers=2)
    assert damaged.read_text() == original
    changed = [p for p in attempts if p.read_text() != attempts[p]]
    assert changed == [damaged.parent / "attempts"]


def test_infrastructure_is_not_skipped(setup):
    root, args = setup
    pl.DataFrame({"id": ["INFRAF"]}).write_parquet(args["metadata"])
    with pytest.raises(RuntimeError, match="infrastructure"):
        export.export_main(**args)
    assert not (root / "export/manifest.json").exists()
    assert not list(root.rglob("state.json"))


def test_input_change_rejected(setup):
    _, args = setup
    export.export_main(**args)
    pl.DataFrame({"id": ["AAAAAA"]}).write_parquet(args["metadata"])
    with pytest.raises(RuntimeError, match="changed"):
        export.export_main(**args)


def test_lock_excludes_other_process(tmp_path):
    lock = tmp_path / "lock"
    with export.exclusive_lock(lock):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import fcntl,sys; f=open(sys.argv[1],'a+'); "
                "fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)",
                str(lock),
            ],
            capture_output=True,
        )
        assert proc.returncode != 0
    with export.exclusive_lock(lock):
        pass


def test_duplicate_records_rejected():
    from conquest_to_parquet import split_cif

    with pytest.raises(ValueError, match="Duplicate"):
        split_cif("data_CSD_CIF_AAAAAA\nx\ndata_CSD_CIF_AAAAAA\nx\n")


def test_interrupted_chunk_resumes_without_repeating_completed_work(setup):
    root, args = setup
    pl.DataFrame({"id": ["AAAAAA", "SLOWWW"]}).write_parquet(args["metadata"])
    args["chunk_size"] = 1
    command = [sys.executable, str(DATA_SCRIPTS / "export_conquest.py")]
    for key, value in args.items():
        command.extend(["--" + key, json.dumps(value) if isinstance(value, list) else str(value)])
    proc = subprocess.Popen(
        command + ["--workers", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + 15
        while not list(root.rglob("slow-once")):
            assert proc.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        pid = int(next(root.rglob("slow-once")).read_text())
        completed = next(root.rglob("state.json"))
        mtime = completed.stat().st_mtime_ns
        proc.terminate()
        assert proc.wait(timeout=10) != 0
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        export.export_main(**args, workers=4)
        assert completed.stat().st_mtime_ns == mtime
        assert (root / "export/manifest.json").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_converter_manifest_and_rejects_corruption(setup):
    from conquest_to_parquet import main as pack

    root, args = setup
    export.export_main(**args)
    manifest = str(root / "export/manifest.json")
    output = root / "packed.parquet"
    pack(manifest=manifest, out=str(output))
    assert pl.read_parquet(output)["id"].to_list() == ["AAAAAA", "CCCCCC"]
    original = output.read_bytes()
    next(root.rglob("export.mol2")).write_text("damaged")
    with pytest.raises(ValueError, match="Damaged"):
        pack(manifest=manifest, out=str(output))
    assert output.read_bytes() == original
    assert not output.with_suffix(".parquet.tmp").exists()


def test_entire_restriction_absent_is_accounted(setup):
    root, args = setup
    pl.DataFrame({"id": ["ABSENT"]}).write_parquet(args["metadata"])
    export.export_main(**args)
    manifest = json.loads((root / "export/manifest.json").read_text())
    assert manifest["chunks"][0]["unmatched"] == ["ABSENT"]
    assert manifest["chunks"][0]["status"] == "complete"
