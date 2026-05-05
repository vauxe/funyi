# coding=utf-8
from __future__ import annotations

import argparse
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command under host-RAM and GPU-VRAM guards."
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Optional working directory for the child process.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=None,
        help="Kill the child process group after this many wall-clock seconds.",
    )
    parser.add_argument(
        "--max-rss-gib",
        type=float,
        default=None,
        help="Kill the child process tree if total RSS exceeds this many GiB.",
    )
    parser.add_argument(
        "--rlimit-as-gib",
        type=float,
        default=None,
        help="Optional RLIMIT_AS hard cap in GiB for the child process.",
    )
    parser.add_argument(
        "--max-vram-mib",
        type=int,
        default=None,
        help="Kill the child process tree if nvidia-smi reports this many MiB or more.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help=(
            "Limit the child to the first N CPUs and set common BLAS/OpenMP "
            "thread env vars to N."
        ),
    )
    parser.add_argument(
        "--cpu-list",
        default=None,
        help="Optional explicit CPU affinity list, for example `0-7,16-23`.",
    )
    parser.add_argument(
        "--nice",
        type=int,
        default=None,
        help="Increase child niceness by this value, for example 10.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variable for the child. May be repeated.",
    )
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=1.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--grace-sec",
        type=float,
        default=5.0,
        help="Grace period after SIGTERM before SIGKILL.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute. Use `--` before the command.",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command; use `-- <cmd> ...`")
    if args.timeout_sec is not None and args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
    if args.max_rss_gib is not None and args.max_rss_gib <= 0:
        parser.error("--max-rss-gib must be positive")
    if args.rlimit_as_gib is not None and args.rlimit_as_gib <= 0:
        parser.error("--rlimit-as-gib must be positive")
    if args.max_vram_mib is not None and args.max_vram_mib <= 0:
        parser.error("--max-vram-mib must be positive")
    if args.cpu_threads is not None and args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if args.nice is not None and args.nice < 0:
        parser.error("--nice must be non-negative")
    if args.poll_sec <= 0:
        parser.error("--poll-sec must be positive")
    if args.grace_sec < 0:
        parser.error("--grace-sec must be non-negative")
    try:
        _cpu_affinity(cpu_threads=args.cpu_threads, cpu_list=args.cpu_list)
        _child_env(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _rss_bytes(pid: int) -> Optional[int]:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return None
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return None


def _read_ppid(pid: int) -> Optional[int]:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return None
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("PPid:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    return None


def _process_tree_pids(root_pid: int) -> set[int]:
    children_by_parent: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        ppid = _read_ppid(pid)
        if ppid is None:
            continue
        children_by_parent.setdefault(ppid, []).append(pid)

    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children_by_parent.get(pid, ()))
    return seen


def _rss_bytes_for_tree(root_pid: int) -> Optional[int]:
    pids = _process_tree_pids(root_pid)
    if not pids:
        return None
    total = 0
    found = False
    for pid in pids:
        rss = _rss_bytes(pid)
        if rss is None:
            continue
        total += rss
        found = True
    return total if found else None


def _gpu_memory_mib(pids: set[int]) -> int:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return 0
    total = 0
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_text, memory_text = [item.strip() for item in line.split(",", 1)]
            if int(pid_text) in pids:
                total += int(memory_text)
        except Exception:
            continue
    return total


def _parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid CPU range: {item}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    if not cpus:
        raise ValueError(f"Invalid CPU list: {value}")
    return cpus


def _allowed_cpus() -> set[int] | None:
    try:
        return set(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _cpu_affinity(*, cpu_threads: Optional[int], cpu_list: Optional[str]) -> set[int] | None:
    allowed = _allowed_cpus()
    if cpu_list:
        affinity = _parse_cpu_list(cpu_list)
    elif cpu_threads is not None:
        base = sorted(allowed) if allowed is not None else list(range(os.cpu_count() or 1))
        affinity = set(base[: max(1, int(cpu_threads))])
    else:
        return None

    if not affinity:
        raise ValueError("CPU affinity cannot be empty")
    if allowed is not None:
        unavailable = sorted(affinity - allowed)
        if unavailable:
            raise ValueError(f"CPU list includes unavailable CPUs: {unavailable}")
    return affinity


def _set_memory_limit(rlimit_as_gib: Optional[float]) -> None:
    if rlimit_as_gib is None:
        return
    limit_bytes = int(float(rlimit_as_gib) * (1024 ** 3))
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def _configure_child(
    *,
    rlimit_as_gib: Optional[float],
    cpu_threads: Optional[int],
    cpu_list: Optional[str],
    nice: Optional[int],
) -> None:
    _set_memory_limit(rlimit_as_gib)
    affinity = _cpu_affinity(cpu_threads=cpu_threads, cpu_list=cpu_list)
    if affinity is not None:
        os.sched_setaffinity(0, affinity)
    if nice is not None and int(nice) != 0:
        os.nice(int(nice))


def _child_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    affinity = _cpu_affinity(cpu_threads=args.cpu_threads, cpu_list=args.cpu_list)
    if args.cpu_threads is not None or affinity is not None:
        thread_count = int(args.cpu_threads) if args.cpu_threads is not None else len(affinity or ())
        if affinity is not None:
            thread_count = min(thread_count, len(affinity))
        threads = str(max(1, thread_count))
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[key] = threads
    for item in args.env:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_process_group_exit(
    proc: subprocess.Popen[object],
    pgid: int,
    *,
    timeout_sec: float,
) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        proc.poll()
        if not _process_group_exists(pgid):
            return True
        time.sleep(0.1)
    proc.poll()
    return not _process_group_exists(pgid)


def _terminate_process_group(proc: subprocess.Popen[object], grace_sec: float) -> None:
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        proc.poll()
        return

    if _wait_process_group_exit(proc, pgid, timeout_sec=grace_sec):
        return

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    _wait_process_group_exit(proc, pgid, timeout_sec=1.0)


def main() -> None:
    args = _parse_args()
    cwd = args.cwd or os.getcwd()
    child_env = _child_env(args)
    proc = subprocess.Popen(
        args.command,
        cwd=cwd,
        env=child_env,
        preexec_fn=(
            lambda: _configure_child(
                rlimit_as_gib=args.rlimit_as_gib,
                cpu_threads=args.cpu_threads,
                cpu_list=args.cpu_list,
                nice=args.nice,
            )
        ),
        start_new_session=True,
    )

    rss_limit = None if args.max_rss_gib is None else int(args.max_rss_gib * (1024 ** 3))
    deadline = None if args.timeout_sec is None else time.time() + float(args.timeout_sec)
    triggered_reason: Optional[str] = None
    try:
        while True:
            returncode = proc.poll()
            if returncode is not None:
                raise SystemExit(returncode)

            tree_pids = _process_tree_pids(proc.pid)

            if deadline is not None and time.time() >= deadline:
                triggered_reason = (
                    f"Wall timeout exceeded: pid={proc.pid} timeout_sec={args.timeout_sec:.1f}"
                )
                break

            if rss_limit is not None:
                rss_bytes = _rss_bytes_for_tree(proc.pid)
                if rss_bytes is not None and rss_bytes >= rss_limit:
                    triggered_reason = (
                        f"RSS limit exceeded: pid={proc.pid} pids={len(tree_pids)} "
                        f"rss_gib={rss_bytes / (1024 ** 3):.2f} "
                        f"limit_gib={args.max_rss_gib:.2f}"
                    )
                    break

            if args.max_vram_mib is not None:
                vram_mib = _gpu_memory_mib(tree_pids)
                if vram_mib >= args.max_vram_mib:
                    triggered_reason = (
                        f"VRAM limit exceeded: pid={proc.pid} pids={len(tree_pids)} "
                        f"vram_mib={vram_mib} "
                        f"limit_mib={args.max_vram_mib}"
                    )
                    break

            time.sleep(args.poll_sec)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        triggered_reason = "Interrupted by user."
    finally:
        if triggered_reason is not None:
            print(triggered_reason, file=sys.stderr)
            _terminate_process_group(proc, args.grace_sec)
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass

    raise SystemExit(125)


if __name__ == "__main__":
    main()
