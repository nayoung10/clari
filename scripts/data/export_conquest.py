# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "jsonargparse"]
# ///
"""Resumable ConQuest export. Run under Xvfb; each worker owns a chunk directory."""

import concurrent.futures as cf
import contextlib
import fcntl
import getpass
import hashlib
import heapq
import json
import os
import pathlib
import re
import select
import signal
import socket
import subprocess
import tempfile
import threading
import time

import polars as pl
from conquest_to_parquet import split_cif, split_mol2
from jsonargparse import auto_cli

QUERY = (
    "any atom\n  query\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.0000    0.0000    0.0000 A   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n"
)
FORMATS = ("cif", "mol2", "gcd")


def digest(path):
    with pathlib.Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = pathlib.Path(path)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


@contextlib.contextmanager
def exclusive_lock(path):
    with pathlib.Path(path).open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another controller holds {path}; stop it before resuming") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"{socket.gethostname()} pid={os.getpid()}\n")
        stream.flush()
        yield


def validate(directory, requested):
    """Validate record boundaries and identities, not scientific preprocessing filters."""
    cifs = split_cif((directory / "export.cif").read_text())
    mol2s = split_mol2((directory / "export.mol2").read_text())
    gcd = (directory / "export.gcd").read_text().split()
    if len(gcd) != len(set(gcd)) or set(cifs) != set(mol2s) or set(cifs) != set(gcd):
        raise ValueError("CIF/MOL2/GCD identifiers differ or repeat")
    if set(gcd) - set(requested):
        raise ValueError("ConQuest returned unrequested identifiers")
    for refcode in gcd:
        # A CIF need not have coordinates: downstream decides scientific eligibility.
        if "_chemical_formula_sum" not in cifs[refcode]:
            raise ValueError(f"Incomplete CIF record for {refcode}")
        lines = mol2s[refcode].splitlines()
        if len(lines) < 3 or not re.match(r"\s*\d+\s+\d+", lines[2]):
            raise ValueError(f"Incomplete MOL2 record for {refcode}")
    return sorted(set(requested) - set(gcd))


class DeferredChunk(RuntimeError):
    def __init__(self, name, retry_at, attempts):
        super().__init__(f"Chunk {name} deferred after {attempts} script-launch failures")
        self.name, self.retry_at, self.attempts = name, retry_at, attempts


class Exporter:
    def __init__(
        self,
        root,
        command,
        databases,
        timeout,
        scratch_dir="/tmp",
        retry_delay=5,
        retry_max_delay=300,
    ):
        self.root, self.command, self.databases = root, command, databases
        self.timeout = timeout
        self.scratch_dir = scratch_dir
        self.retry_delay, self.retry_max_delay = retry_delay, retry_max_delay
        self.stop = threading.Event()
        self.launched = 0
        self.fresh_accounted = 0
        self.counter_lock = threading.Lock()
        # Read the existing upstream split definition without importing the ML environment.
        import runpy

        csd = pathlib.Path(__file__).resolve().parents[2] / "clari" / "csd.py"
        self.test_families = set(runpy.run_path(str(csd))["AVAILABLE_CSD_SUBSETS"]["test"])

    def invoke(self, directory, ids):
        retry_path = directory / "retry.json"
        retry = json.loads(retry_path.read_text()) if retry_path.exists() else {}
        if retry.get("retry_at", 0) > time.time():
            raise DeferredChunk(directory.name, retry["retry_at"], retry["attempts"])
        # Keep only disposable ConQuest scratch on the node. Outputs/checkpoints stay shared.
        with tempfile.TemporaryDirectory(prefix="clari-conquest-", dir=self.scratch_dir) as scratch:
            result = self.invoke_attempt(directory, ids, scratch)
        if result["status"] == "permission_retry":
            attempts = retry.get("attempts", 0) + 1
            delay = min(self.retry_max_delay, self.retry_delay * 2 ** min(attempts - 1, 16))
            retry = {
                "attempts": attempts,
                "retry_at": time.time() + delay,
                "error": result["error"],
            }
            # Retain the latest failure log and durable retry history without unbounded log growth.
            (directory / "console.log").replace(directory / "permission-failure.log")
            atomic_json(retry_path, retry)
            raise DeferredChunk(directory.name, retry["retry_at"], attempts)
        retry_path.unlink(missing_ok=True)
        return result

    def invoke_attempt(self, directory, ids, scratch):
        if self.stop.is_set():
            raise InterruptedError("Export stopped; completed chunks are retained")
        for ext in (*FORMATS, "cqs"):
            (directory / f"export.{ext}").unlink(missing_ok=True)
        (directory / "ids.gcd").write_text("\n".join(ids) + "\n")
        command = [
            self.command,
            "-j",
            "export",
            # ConQuest otherwise shares ~/csds_data/searches/batch27 across processes.
            "-user-directory",
            scratch,
            "-db",
            *self.databases,
            "-require",
            str(self.root / "any.mol"),
            "-restrict",
            str(directory / "ids.gcd"),
            "-export",
            *FORMATS,
            "-log",
            "cq.log",
        ]
        with self.counter_lock:
            self.launched += 1
        started = time.monotonic()
        with (directory / "console.log").open("w") as output:
            process = subprocess.Popen(
                command,
                cwd=directory,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while process.poll() is None:
                    if self.stop.wait(0.2):
                        raise InterruptedError("Export interrupted")
                    if time.monotonic() - started > self.timeout:
                        raise RuntimeError(f"ConQuest timed out; inspect {directory}/console.log")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        log = (directory / "console.log").read_text(errors="replace")
        missing = set(re.findall(r"^WARNING: (\S+) not found$", log, re.M))
        empty_restriction = (
            process.returncode > 0
            and "ERROR: No entries found in restriction file " in log
            and missing == set(ids)
        )
        if empty_restriction:
            # ConQuest exits 1 and writes no exports when every listed ID is absent.
            for ext in FORMATS:
                (directory / f"export.{ext}").write_text("")
        if process.returncode == 0 or empty_restriction:
            try:
                unmatched = validate(directory, ids)
            except (OSError, ValueError) as exc:
                # Zero exit alone is not success; do not turn an unknown failure into data loss.
                raise RuntimeError(f"Invalid export in {directory}: {exc}") from exc
            return {
                "status": "complete",
                "unmatched": unmatched,
                "hashes": {ext: digest(directory / f"export.{ext}") for ext in FORMATS},
            }
        # A generated search-script launch failure is infrastructure, never a bad entry.
        if (
            "PermissionError: [Errno 13] Permission denied:" in log
            and "create_and_run_thomas_script" in log
            and re.search(r"Permission denied: .*[/\\]searches[/\\].*\.sh['\"]", log)
        ):
            return {"status": "permission_retry", "error": log[-5000:]}
        # Only a traceback inside the entry exporter is eligible for bisection/skipping.
        # License, display, database, OOM, timeout, and disk failures abort instead.
        infrastructure = re.search(
            r"no space left|disk quota|permission denied|input/output error|"
            r"out of memory|cannot allocate memory|couldn.t connect to display|"
            r"no display name|licen[cs]e.*(?:fail|invalid|expired|unavailable)",
            log,
            re.I,
        )
        if (
            process.returncode < 0
            or infrastructure
            or "Traceback (most recent call last)" not in log
            or "export_filters/" not in log
            or "_export_entry" not in log
        ):
            raise RuntimeError(
                f"ConQuest infrastructure/unknown failure; inspect {directory}/console.log"
            )
        return {"status": "failed", "error": re.sub(r"\x1b\[[0-9;]*m", "", log[-5000:])}

    def chunk(self, name, ids):
        directory = self.root / "chunks" / name
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state["status"] == "complete":
            if all(
                (directory / f"export.{ext}").is_file()
                and digest(directory / f"export.{ext}") == state["hashes"][ext]
                for ext in FORMATS
            ):
                return self.leaf(name, ids, state)
            print(f"Re-exporting damaged checkpoint {name}", flush=True)
            state = None
        if state and state["status"] == "skipped":
            return self.leaf(name, ids, state)
        if not state or state["status"] != "split":
            state = self.invoke(directory, ids)
            if state["status"] == "failed" and len(ids) == 1:
                (directory / "console.log").replace(directory / "first-failure.log")
                state = self.invoke(directory, ids)
                if state["status"] == "failed":
                    state.update(
                        status="skipped", id=ids[0], test_family=ids[0][:6] in self.test_families
                    )
            if state["status"] == "failed":
                state = {"status": "split"}
            atomic_json(state_path, state)
            if state["status"] != "split":
                with self.counter_lock:
                    self.fresh_accounted += len(ids)
        if state["status"] != "split":
            return self.leaf(name, ids, state)
        # Persist the split before visiting children, so restart never repeats the failed parent.
        for ext in (*FORMATS, "cqs"):
            (directory / f"export.{ext}").unlink(missing_ok=True)
        mid = len(ids) // 2
        return self.chunk(name + "0", ids[:mid]) + self.chunk(name + "1", ids[mid:])

    @staticmethod
    def leaf(name, ids, state):
        return [{"directory": f"chunks/{name}", "requested": len(ids), **state}]


def export_main(
    metadata: str,
    out: str,
    cqbatch: str,
    databases: list[str],
    workers: int = 4,
    chunk_size: int = 1000,
    timeout: int = 1800,
    scratch_dir: str = "/tmp",
    retry_delay: float = 5,
    retry_max_delay: float = 300,
):
    if (
        workers < 1
        or chunk_size < 1
        or timeout < 1
        or retry_delay <= 0
        or retry_max_delay < retry_delay
    ):
        raise ValueError("workers, chunk_size, and timeout must be positive")
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("ConQuest restriction needs DISPLAY; run this command under Xvfb")
    os.environ.setdefault("LOGNAME", getpass.getuser())
    root = pathlib.Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    command = str(pathlib.Path(cqbatch).resolve())
    databases = [str(pathlib.Path(db).resolve()) for db in databases]
    ids = sorted(pl.read_parquet(metadata, columns=["id"])["id"].to_list())
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(not re.fullmatch(r"[A-Z]{6}\d*", x) for x in ids)
    ):
        raise ValueError("Metadata must contain unique CSD refcodes")
    version = subprocess.run(
        [command, "-j", "version", "-version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()
    db_info = []
    for db in databases:
        path = pathlib.Path(db + "_CIP.sqlite")
        stat = path.stat()
        db_info.append(
            {
                "name": pathlib.Path(db).name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "info_sha256": digest(db + ".inf"),
            }
        )
    config = {
        "schema": 1,
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "count": len(ids),
        "chunk_size": chunk_size,
        "databases": db_info,
        "conquest_version": version,
        "query": QUERY,
        "formats": FORMATS,
    }
    # JSON roundtrip normalizes the tuple for comparison with disk.
    config = json.loads(json.dumps(config))
    with exclusive_lock(root / ".controller.lock"):
        config_path = root / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise RuntimeError("Export inputs/version/options changed; use a new output directory")
        atomic_json(config_path, config)
        (root / "any.mol").write_text(QUERY)
        exporter = Exporter(
            root, command, databases, timeout, scratch_dir, retry_delay, retry_max_delay
        )
        old_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, lambda *_: exporter.stop.set())
        started, accounted, leaves = time.monotonic(), 0, []
        try:
            # Deferred chunks release their worker slot; healthy chunks continue immediately.
            pending = [
                (0, i, f"{i // chunk_size:06d}-", ids[i : i + chunk_size])
                for i in range(0, len(ids), chunk_size)
            ]
            heapq.heapify(pending)
            serial = len(ids)
            active = {}
            failures, probe_only, blocked_until = 0, False, 0
            heartbeat = time.monotonic()
            with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                try:
                    while pending or active:
                        if exporter.stop.is_set():
                            raise InterruptedError("Export stopped; completed chunks are retained")
                        now = time.time()
                        limit = 1 if probe_only else workers
                        while (
                            pending
                            and len(active) < limit
                            and now >= blocked_until
                            and pending[0][0] <= now
                        ):
                            _, _, name, chunk_ids = heapq.heappop(pending)
                            active[pool.submit(exporter.chunk, name, chunk_ids)] = (name, chunk_ids)
                        done, _ = (
                            cf.wait(active, timeout=0.2, return_when=cf.FIRST_COMPLETED)
                            if active
                            else (set(), set())
                        )
                        if not active and not done:
                            exporter.stop.wait(0.2)
                        for future in done:
                            name, chunk_ids = active.pop(future)
                            try:
                                result = future.result()
                            except DeferredChunk as exc:
                                failures += 1
                                serial += 1
                                heapq.heappush(pending, (exc.retry_at, serial, name, chunk_ids))
                                print(
                                    f"Deferred {exc.name}: script permission failure #{exc.attempts}; "
                                    f"retry in {max(0, exc.retry_at - time.time()):.1f}s",
                                    flush=True,
                                )
                                if failures >= 8:
                                    probe_only = True
                                    blocked_until = max(blocked_until, exc.retry_at)
                                continue
                            failures, probe_only, blocked_until = 0, False, 0
                            leaves.extend(result)
                            accounted += sum(leaf["requested"] for leaf in result)
                            elapsed = time.monotonic() - started
                            fresh = exporter.fresh_accounted
                            eta = (
                                ((len(ids) - accounted) * elapsed / fresh / 3600) if fresh else None
                            )
                            eta_text = f"{eta:.2f}h" if eta is not None else "waiting for new work"
                            print(
                                f"Accounted {accounted:,}/{len(ids):,}; elapsed {elapsed:.0f}s; "
                                f"ETA {eta_text}",
                                flush=True,
                            )
                        if time.monotonic() - heartbeat >= 30:
                            print(
                                f"Export alive: {accounted:,}/{len(ids):,} accounted; "
                                f"{len(active)} running, {len(pending)} pending; "
                                f"recovery probe mode={probe_only}",
                                flush=True,
                            )
                            heartbeat = time.monotonic()
                except BaseException:
                    exporter.stop.set()
                    for future in active:
                        future.cancel()
                    raise
            if exporter.stop.is_set():
                raise InterruptedError("Export stopped")
            leaves.sort(key=lambda leaf: leaf["directory"])
            manifest = {"config": config, "requested": len(ids), "chunks": leaves}
            atomic_json(root / "manifest.json", manifest)
            skipped = [leaf for leaf in leaves if leaf["status"] == "skipped"]
            atomic_json(root / "skipped.json", skipped)
            unmatched = sum(len(leaf.get("unmatched", [])) for leaf in leaves)
            print(
                f"Export complete: {len(ids) - len(skipped) - unmatched:,} exported, "
                f"{len(skipped)} skipped, {unmatched} unmatched; "
                f"{exporter.launched} ConQuest jobs launched",
                flush=True,
            )
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


@contextlib.contextmanager
def virtual_display(executable, root):
    if os.environ.get("DISPLAY"):
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / "xvfb.log").open("w") as log:
        server = subprocess.Popen(
            [executable, "-displayfd", "1", "-screen", "0", "800x600x24", "-nolisten", "tcp"],
            stdout=subprocess.PIPE,
            stderr=log,
        )
        try:
            if not select.select([server.stdout], [], [], 20)[0]:
                raise RuntimeError(f"Xvfb startup timed out; inspect {root}/xvfb.log")
            number = server.stdout.readline().decode().strip()
            if not number.isdigit():
                raise RuntimeError(f"Xvfb failed; inspect {root}/xvfb.log")
            os.environ["DISPLAY"] = ":" + number
            yield
        finally:
            os.environ.pop("DISPLAY", None)
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            server.stdout.close()


def main(
    metadata: str,
    out: str,
    cqbatch: str,
    databases: list[str],
    workers: int = 4,
    chunk_size: int = 1000,
    timeout: int = 1800,
    xvfb: str = "Xvfb",
    scratch_dir: str = "/tmp",
    retry_delay: float = 5,
    retry_max_delay: float = 300,
):
    with virtual_display(xvfb, pathlib.Path(out).resolve()):
        export_main(
            metadata,
            out,
            cqbatch,
            databases,
            workers,
            chunk_size,
            timeout,
            scratch_dir,
            retry_delay,
            retry_max_delay,
        )


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
