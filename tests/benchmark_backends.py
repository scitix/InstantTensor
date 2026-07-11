#!/usr/bin/env python3
"""Run tests/test.py against multiple backend/package versions.

Edit tests/benchmark_backends.json, then run this file from anywhere in the repository:

    python tests/benchmark_backends.py

Backend specs:
  - pypi/<backend>/<version>, for example pypi/safetensors/0.7.0
  - pypi/<backend>, for example pypi/safetensors
  - local, meaning the current repository commit
  - local/<git-ref>, for example local/main or local/v0.1.9

For local and local/<git-ref>, the backend is instanttensor and the package is
installed from a temporary clean copy of this repository after checking out the ref.
For instanttensor backend, only local specs are supported.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


DEFAULT_CONFIG = Path(__file__).with_name("benchmark_backends.json")
DEFAULT_PYTHON_VERSION = "3.12"


VALID_BACKENDS = {
    "instanttensor",
    "safetensors",
    "runai_model_streamer",
    "fastsafetensors",
}

PYPI_PACKAGE_NAMES = {
    "instanttensor": "instanttensor",
    "safetensors": "safetensors",
    "runai_model_streamer": "runai-model-streamer",
    "fastsafetensors": "fastsafetensors",
}

# Add custom indexes or constraints here, or set INSTANTTENSOR_BENCH_PIP_ARGS.
# Example: EXTRA_PIP_ARGS = ["--index-url", "https://download.pytorch.org/whl/cu128"]
EXTRA_PIP_ARGS: list[str] = []
SUBMODULE_JOBS = "16"

BOOST_174_SUBMODULES = [
    "lockfree",
    "align",
    "array",
    "assert",
    "atomic",
    "config",
    "core",
    "integer",
    "iterator",
    "mpl",
    "parameter",
    "predef",
    "static_assert",
    "tuple",
    "type_traits",
    "utility",
    "winapi",
    "concept_check",
    "mp11",
    "conversion",
    "typeof",
    "move",
    "detail",
    "function_types",
    "fusion",
    "optional",
    "smart_ptr",
    "container_hash",
    "io",
    "preprocessor",
    "throw_exception",
]
LEGACY_BOOST_SUBMODULE_PATH = "csrc/third_party/boost"


class BenchError(RuntimeError):
    pass


GREEN = "\033[32m"
RESET = "\033[0m"


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def bench_print(*args: object, file: Any | None = None, **kwargs: Any) -> None:
    stream = file if file is not None else sys.stdout
    text = " ".join(str(arg) for arg in args)
    print(f"{GREEN}{text}{RESET}", file=stream, **kwargs)


def run_streamed(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
    dry_run: bool = False,
) -> None:
    printable = shlex.join(cmd)
    if env is not None:
        env_overrides = {key: value for key, value in env.items() if os.environ.get(key) != value}
        if env_overrides:
            printable = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in sorted(env_overrides.items())
            ) + f" {printable}"
    prefix = f"(cd {cwd} && {printable})" if cwd else printable
    bench_print(f"\n$ {prefix}", flush=True)
    if dry_run:
        return

    log_handle = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("a", encoding="utf-8")
        log_handle.write(f"\n$ {prefix}\n")
        log_handle.flush()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            if log_handle is not None:
                log_handle.write(line)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    finally:
        if log_handle is not None:
            log_handle.close()


def output_text(cmd: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def repo_root() -> Path:
    return Path(output_text(["git", "rev-parse", "--show-toplevel"])).resolve()


def parse_backend_spec(spec: str) -> dict[str, str | None]:
    if spec == "local" or spec.startswith("local/"):
        ref = "HEAD" if spec == "local" else spec[len("local/") :]
        if not ref:
            raise BenchError(f"Invalid backend spec {spec!r}: missing git ref")
        return {
            "source": "local",
            "backend": "instanttensor",
            "package": "instanttensor",
            "version": None,
            "ref": ref,
        }

    parts = spec.split("/")
    if len(parts) not in (2, 3) or parts[0] != "pypi":
        raise BenchError(
            f"Invalid backend spec {spec!r}; expected pypi/<backend>[/<version>], local, or local/<git-ref>"
        )

    backend = parts[1]
    if backend == "instanttensor":
        raise BenchError(
            "InstantTensor backend no longer supports pypi installs. Use local/<git-ref> instead."
        )
    if backend not in VALID_BACKENDS:
        raise BenchError(
            f"Invalid backend {backend!r}; expected one of {sorted(VALID_BACKENDS)}"
        )

    version = parts[2] if len(parts) == 3 else None
    return {
        "source": "pypi",
        "backend": backend,
        "package": PYPI_PACKAGE_NAMES[backend],
        "version": version,
        "ref": None,
    }


def package_requirement(package: str, version: str | None) -> str:
    return f"{package}=={version}" if version else package


def instanttensor_requirement(version: str | None) -> str:
    return package_requirement("instanttensor[test]", version)


def env_python_path(env_dir: Path) -> Path:
    suffix = "python.exe" if os.name == "nt" else "bin/python"
    return env_dir / suffix


def create_conda_env(
    python_version: str,
    env_dir: Path,
    *,
    reuse_env: bool,
    log_file: Path | None = None,
    dry_run: bool,
) -> Path:
    env_python = env_python_path(env_dir)
    if reuse_env and env_dir.exists() and env_python.exists():
        bench_print(f"Reusing conda env: {env_dir}", flush=True)
        return env_python

    if env_dir.exists() and not dry_run:
        shutil.rmtree(env_dir)

    run_streamed(
        ["conda", "create", "-y", "--quiet", "-p", str(env_dir), f"python={python_version}"],
        log_file=log_file,
        dry_run=dry_run,
    )
    return env_python


def clone_conda_env(
    source_env: Path,
    target_env: Path,
    *,
    reuse_env: bool,
    log_file: Path | None = None,
    dry_run: bool = False,
) -> Path:
    env_python = env_python_path(target_env)
    if reuse_env and target_env.exists() and env_python.exists():
        bench_print(f"Reusing cloned env: {target_env}", flush=True)
        return env_python

    if target_env.exists() and not dry_run:
        shutil.rmtree(target_env)

    run_streamed(
        [
            "conda",
            "create",
            "--quiet",
            "-y",
            "--clone",
            str(source_env),
            "-p",
            str(target_env),
        ],
        log_file=log_file,
        dry_run=dry_run,
    )
    return env_python


def parse_freeze_line(line: str) -> tuple[str, str] | None:
    cleaned = line.strip()
    if not cleaned or cleaned.startswith("-e "):
        return None
    if " #" in cleaned:
        cleaned = cleaned.split(" #", 1)[0].rstrip()
    cleaned = cleaned.split(";", 1)[0].strip()
    if not cleaned:
        return None

    if " @ " in cleaned:
        raw_name = cleaned.split(" @ ", 1)[0]
    elif "==" in cleaned:
        raw_name = cleaned.split("==", 1)[0]
    else:
        return None

    raw_name = raw_name.split("[", 1)[0]
    name = normalize_package_name(raw_name)
    if not name:
        return None
    return name, cleaned


def capture_env_snapshot(env_python: Path, snapshot_file: Path, *, dry_run: bool) -> list[str]:
    if dry_run:
        return []
    snapshot = output_text([str(env_python), "-m", "pip", "freeze"]).splitlines()
    snapshot_file.write_text("\n".join(snapshot), encoding="utf-8")
    return snapshot


def pip_install(
    env_python: Path,
    requirements: list[str],
    *,
    cwd: Path | None = None,
    log_file: Path | None = None,
    dry_run: bool = False,
) -> None:
    extra_args = EXTRA_PIP_ARGS + shlex.split(
        os.environ.get("INSTANTTENSOR_BENCH_PIP_ARGS", "")
    )
    cmd = [str(env_python), "-m", "pip", "install", "--quiet", *extra_args, *requirements]
    run_streamed(cmd, cwd=cwd, log_file=log_file, dry_run=dry_run)


def install_torch(env_python: Path, torch_version: str | None, *, log_file: Path | None, dry_run: bool) -> None:
    pip_install(
        env_python,
        [package_requirement("torch", torch_version)],
        log_file=log_file,
        dry_run=dry_run,
    )


def restore_env_snapshot(
    env_python: Path,
    snapshot: list[str],
    snapshot_file: Path,
    *,
    log_file: Path | None = None,
    dry_run: bool = False,
) -> None:
    if dry_run:
        return

    desired: dict[str, str] = {}
    for line in snapshot:
        parsed = parse_freeze_line(line)
        if parsed is not None:
            name, spec = parsed
            desired[name] = spec

    current = output_text([str(env_python), "-m", "pip", "freeze"]).splitlines()
    remove: list[str] = []
    for line in current:
        parsed = parse_freeze_line(line)
        if parsed is None:
            continue
        name, _ = parsed
        if name not in desired:
            remove.append(name)

    if remove:
        run_streamed(
            [str(env_python), "-m", "pip", "uninstall", "-y", *remove],
            log_file=log_file,
            dry_run=dry_run,
        )

    if snapshot:
        run_streamed(
            [
                str(env_python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "-r",
                str(snapshot_file),
            ],
            log_file=log_file,
            dry_run=dry_run,
        )


def gitmodules_has_submodule_path(source_dir: Path, submodule_path: str) -> bool:
    gitmodules = source_dir / ".gitmodules"
    if not gitmodules.is_file():
        return False

    try:
        paths = output_text(
            [
                "git",
                "config",
                "--file",
                str(gitmodules),
                "--get-regexp",
                r"^submodule\..*\.path$",
            ],
            cwd=source_dir,
        )
    except subprocess.CalledProcessError:
        return False

    for line in paths.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1] == submodule_path:
            return True
    return False


def initialize_local_submodules(source_dir: Path, *, log_file: Path | None) -> None:
    checkout_script = source_dir / "checkout_submodules.sh"
    if checkout_script.is_file():
        run_streamed(["bash", str(checkout_script)], cwd=source_dir, log_file=log_file)
        return

    has_legacy_boost = gitmodules_has_submodule_path(source_dir, LEGACY_BOOST_SUBMODULE_PATH)
    # v0.1.6 and older use Boost 1.74 as a top-level submodule with selected nested libs.
    if has_legacy_boost:
        run_streamed(["git", "submodule", "sync"], cwd=source_dir, log_file=log_file)
        run_streamed(
            ["git", "submodule", "update", "--init", "--jobs", SUBMODULE_JOBS],
            cwd=source_dir,
            log_file=log_file,
        )

        boost_libs = source_dir / LEGACY_BOOST_SUBMODULE_PATH / "libs"
        if not boost_libs.is_dir():
            raise BenchError(f"Legacy Boost submodule was not initialized: {boost_libs}")

        run_streamed(["git", "submodule", "sync"], cwd=boost_libs, log_file=log_file)
        run_streamed(
            ["git", "submodule", "update", "--init", "--jobs", SUBMODULE_JOBS, *BOOST_174_SUBMODULES],
            cwd=boost_libs,
            log_file=log_file,
        )
        return

    run_streamed(["git", "submodule", "sync", "--recursive"], cwd=source_dir, log_file=log_file)
    run_streamed(
        ["git", "submodule", "update", "--init", "--recursive", "--jobs", SUBMODULE_JOBS],
        cwd=source_dir,
        log_file=log_file,
    )


def prepare_local_source(
    root: Path,
    ref: str,
    entry_dir: Path,
    *,
    log_file: Path | None,
    dry_run: bool,
) -> Path:
    source_dir = entry_dir / "source"
    if dry_run:
        bench_print(f"Would copy {root} to {source_dir} and checkout {ref!r}")
        return source_dir

    try:
        source_dir.resolve().relative_to(root.resolve())
    except ValueError:
        pass
    else:
        raise BenchError(
            "For local/... entries, choose a --work-dir outside the repository"
        )

    if source_dir.exists():
        shutil.rmtree(source_dir)
    shutil.copytree(root, source_dir, symlinks=True)
    run_streamed(["git", "submodule", "deinit", "-f", "--all"], cwd=source_dir, log_file=log_file)
    run_streamed(["git", "clean", "-ffdx"], cwd=source_dir, log_file=log_file)
    run_streamed(["git", "checkout", "--force", ref], cwd=source_dir, log_file=log_file)
    run_streamed(["git", "clean", "-ffdx"], cwd=source_dir, log_file=log_file)
    initialize_local_submodules(source_dir, log_file=log_file)
    run_streamed(["git", "clean", "-ffdx"], cwd=source_dir, log_file=log_file)

    status = output_text(["git", "status", "--porcelain"], cwd=source_dir)
    if status:
        raise BenchError(f"Local source copy is not clean before install:\n{status}")

    return source_dir


def install_backend(
    spec: dict[str, str | None],
    env_python: Path,
    root: Path,
    entry_dir: Path,
    *,
    log_file: Path | None,
    dry_run: bool,
) -> Path | None:
    if spec["source"] == "pypi":
        requirements = [
            instanttensor_requirement(None),
            package_requirement(str(spec["package"]), spec["version"]),
        ]
        pip_install(env_python, requirements, log_file=log_file, dry_run=dry_run)
        return None

    source_dir = prepare_local_source(
        root,
        str(spec["ref"]),
        entry_dir,
        log_file=log_file,
        dry_run=dry_run,
    )
    pip_install(
        env_python,
        [".[test]"],
        cwd=source_dir,
        log_file=log_file,
        dry_run=dry_run,
    )
    return source_dir


def expand_model_files(model: str, max_files: int | None) -> list[str]:
    model_path = Path(model).expanduser()
    if model_path.is_dir():
        files = [str(p) for p in model_path.glob("*.safetensors")]
    elif any(ch in model for ch in "*?[]"):
        files = glob.glob(os.path.expanduser(model))
    elif model_path.is_file():
        files = [str(model_path)]
    else:
        raise BenchError(
            f"Model path {model!r} does not exist and is not a glob with matches"
        )

    files = sorted(files)
    if not files:
        raise BenchError(f"No safetensors files found for model {model!r}")

    if max_files is not None:
        if max_files <= 0:
            raise BenchError(f"max_files must be positive, got {max_files}")
        files = files[:max_files]
    return files


def model_cache_targets(model: str, files: list[str]) -> list[Path]:
    model_path = Path(model).expanduser()
    if model_path.is_dir():
        return [model_path]

    targets: list[Path] = []
    seen: set[Path] = set()
    for file in files:
        parent = Path(file).expanduser().parent
        if parent not in seen:
            seen.add(parent)
            targets.append(parent)
    return targets


def evict_model_page_cache(
    model: str,
    files: list[str],
    *,
    log_file: Path | None,
    dry_run: bool,
) -> None:
    targets = model_cache_targets(model, files)
    if not targets:
        return
    run_streamed(
        ["vmtouch", "-e", *[str(target) for target in targets]],
        log_file=log_file,
        dry_run=dry_run,
    )


def case_nproc(case: dict[str, Any]) -> int:
    if "nproc_per_node" in case:
        return int(case["nproc_per_node"])

    args = [str(arg) for arg in case.get("args", [])]
    values: dict[str, int] = {"--tp": 1, "--pp": 1}
    for flag in values:
        if flag in args:
            idx = args.index(flag)
            if idx + 1 >= len(args):
                raise BenchError(f"{flag} is missing its value in case args")
            values[flag] = int(args[idx + 1])
    return values["--tp"] * values["--pp"]


def run_case(
    *,
    env_python: Path,
    test_py: Path,
    backend: str,
    case: dict[str, Any],
    run_dir: Path,
    env_info: str,
    log_file: Path | None,
    dry_run: bool,
) -> None:
    if "model" not in case:
        raise BenchError(f"Case is missing required 'model': {case!r}")

    model = str(case["model"])
    files = expand_model_files(model, case.get("max_files"))
    args = [str(arg) for arg in case.get("args", [])]
    pre_evict_page_cache = case.get("pre_evict_page_cache", True)
    post_evict_page_cache = case.get("post_evict_page_cache", True)

    if not isinstance(pre_evict_page_cache, bool):
        raise BenchError("case field 'pre_evict_page_cache' must be a boolean")
    if not isinstance(post_evict_page_cache, bool):
        raise BenchError("case field 'post_evict_page_cache' must be a boolean")
    nproc = case_nproc(case)

    test_env = os.environ.copy()
    if backend == "instanttensor":
        test_env.setdefault("INSTANTTENSOR_DEBUG", "1")

    case_envs = case.get("envs", {})
    if not isinstance(case_envs, dict):
        raise BenchError("case field 'envs' must be an object")
    normalized_envs: dict[str, str] = {}
    for key, value in case_envs.items():
        normalized_envs[str(key)] = str(value)
        test_env[str(key)] = str(value)

    if nproc > 1:
        cmd = [
            str(env_python),
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={nproc}",
            str(test_py),
            backend,
            *files,
            *args,
        ]
    else:
        cmd = [str(env_python), str(test_py), backend, *files, *args]

    bench_print(
        f"\n=== backend={backend} model={case['model']} files={len(files)} nproc={nproc} args={args} envs={json.dumps(normalized_envs, sort_keys=True)} backend_env={env_info} ===",
        flush=True,
    )
    if pre_evict_page_cache:
        evict_model_page_cache(model, files, log_file=log_file, dry_run=dry_run)
    try:
        run_streamed(cmd, cwd=run_dir, env=test_env, log_file=log_file, dry_run=dry_run)
    finally:
        if post_evict_page_cache:
            evict_model_page_cache(model, files, log_file=log_file, dry_run=dry_run)


def stable_work_dir() -> Path:
    return Path(tempfile.gettempdir()) / "instanttensor-bench"


def entry_dir_name(entry_index: int, backend_spec: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in backend_spec).strip("-")
    safe = safe[:48] or "backend"
    return f"entry-{entry_index}-{safe}"


def load_entries(config_path: Path) -> list[dict[str, Any]]:
    if not config_path.is_file():
        raise BenchError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and isinstance(data.get("entries"), list):
        entries = data["entries"]
    else:
        raise BenchError(
            "Config must be either a JSON list or an object with an 'entries' list"
        )

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise BenchError(f"Entry {idx} must be a JSON object")

    return entries


def selected_entries(entries: list[dict[str, Any]], indexes: list[int] | None) -> list[tuple[int, dict[str, Any]]]:
    if indexes is None:
        selected = list(enumerate(entries))
    else:
        selected = []
        for idx in indexes:
            if idx < 0 or idx >= len(entries):
                raise BenchError(f"Entry index {idx} is out of range 0..{len(entries) - 1}")
            selected.append((idx, entries[idx]))

    return [(idx, entry) for idx, entry in selected if entry.get("enabled", True)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark tests/test.py across backend package versions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON config file with an entries list. Defaults to tests/benchmark_backends.json.",
    )
    parser.add_argument(
        "--entry",
        type=int,
        action="append",
        help="Only run the entry at this zero-based config index. Can be repeated.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Directory for conda envs, source copies, and logs. Defaults to a "
            "stable temp dir outside the repository."
        ),
    )
    parser.add_argument(
        "--python-version",
        default=DEFAULT_PYTHON_VERSION,
        help="Python version passed to conda create. Defaults to 3.12.",
    )
    parser.add_argument(
        "--torch-version",
        default=None,
        help=(
            "Install this torch version into the shared base env before installing "
            "benchmark packages. If omitted, install unpinned torch."
        ),
    )
    parser.add_argument(
        "--backend-env-mode",
        choices=("clone", "inplace"),
        default="clone",
        help=(
            "Backend env mode: 'clone' clones a shared base environment per entry, "
            "or 'inplace' installs directly into the base environment and restores it after each entry."
        ),
    )
    parser.add_argument(
        "--keep-envs",
        action="store_true",
        help="Keep conda environments and local source copies after the run.",
    )
    parser.add_argument(
        "--reuse-envs",
        action="store_true",
        help="Reuse existing conda envs and always rerun pip install.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later entries/cases after a failure.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print install and test commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    test_py = root / "tests" / "test.py"
    if not test_py.is_file():
        raise BenchError(f"Cannot find {test_py}")

    config_path = args.config.expanduser().resolve()
    entries = selected_entries(load_entries(config_path), args.entry)
    if not entries:
        bench_print(f"No enabled benchmark entries configured in {config_path}.")
        return 0

    base_tmp = args.work_dir.expanduser().resolve() if args.work_dir else stable_work_dir()
    base_tmp.mkdir(parents=True, exist_ok=True)

    bench_print(f"Repository: {root}")
    bench_print(f"Config: {config_path}")
    bench_print(f"Work dir: {base_tmp}")
    bench_print(f"Test script: {test_py}")
    bench_print(f"Backend env mode: {args.backend_env_mode}")

    base_env_dir = base_tmp / "base-env"
    base_log = base_tmp / "base.log"
    base_env_python = create_conda_env(
        args.python_version,
        base_env_dir,
        reuse_env=args.reuse_envs,
        log_file=base_log,
        dry_run=args.dry_run,
    )
    install_torch(
        base_env_python,
        args.torch_version,
        log_file=base_log,
        dry_run=args.dry_run,
    )

    base_snapshot: list[str] = []
    base_snapshot_file = base_tmp / "base_snapshot.txt"
    if args.backend_env_mode == "inplace":
        base_snapshot = capture_env_snapshot(
            base_env_python, base_snapshot_file, dry_run=args.dry_run
        )

    failures: list[str] = []
    for entry_index, entry in entries:
        if "backend" not in entry:
            raise BenchError(f"Entry {entry_index} is missing required 'backend'")
        if "cases" not in entry:
            raise BenchError(f"Entry {entry_index} is missing required 'cases'")

        backend_spec = str(entry["backend"])
        spec = parse_backend_spec(backend_spec)
        entry_dir = base_tmp / entry_dir_name(entry_index, backend_spec)
        env_dir = entry_dir / "env"
        run_dir = entry_dir / "run"
        log_file = entry_dir / "benchmark.log"
        run_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        try:
            bench_print(f"\n### Entry {entry_index}: {entry['backend']} ###", flush=True)

            if args.backend_env_mode == "clone":
                env_python = clone_conda_env(
                    base_env_dir,
                    env_dir,
                    reuse_env=args.reuse_envs,
                    log_file=log_file,
                    dry_run=args.dry_run,
                )
                env_info = str(env_dir)
            else:
                env_python = base_env_python
                env_info = f"{base_env_python} (inplace)"

            installed_source = install_backend(
                spec,
                env_python,
                root,
                entry_dir,
                log_file=log_file,
                dry_run=args.dry_run,
            )

            if spec["source"] == "local":
                local_source = entry_dir / "source"
                if installed_source is not None:
                    local_source = installed_source
                if not args.dry_run and not local_source.is_dir():
                    raise BenchError(f"Local source checkout is missing: {local_source}")
                entry_test_py = local_source / "tests" / "test.py"
                if not args.dry_run and not entry_test_py.is_file():
                    raise BenchError(f"Cannot find test script in local checkout: {entry_test_py}")
            elif spec["source"] == "pypi":
                entry_test_py = test_py
            else:
                raise BenchError(f"Unsupported backend source: {spec['source']}")

            for case in entry["cases"]:
                run_case(
                    env_python=env_python,
                    test_py=entry_test_py,
                    backend=str(spec["backend"]),
                    case=case,
                    run_dir=run_dir,
                    env_info=env_info,
                    log_file=log_file,
                    dry_run=args.dry_run,
                )
        except Exception as exc:
            failures.append(f"entry {entry_index} ({entry['backend']}): {exc}")
            bench_print(f"\nFAILED: {failures[-1]}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break
        finally:
            if args.backend_env_mode == "inplace" and not args.dry_run:
                restore_env_snapshot(
                    base_env_python,
                    base_snapshot,
                    base_snapshot_file,
                    log_file=base_log,
                    dry_run=args.dry_run,
                )
            elapsed = time.perf_counter() - started
            bench_print(f"\nEntry {entry_index} finished in {elapsed:.1f}s")
            if not args.keep_envs and not args.dry_run:
                if args.backend_env_mode == "inplace" or not args.reuse_envs:
                    shutil.rmtree(entry_dir, ignore_errors=True)

    if not (args.keep_envs or args.reuse_envs) and not args.dry_run and base_tmp.exists():
        shutil.rmtree(base_env_dir, ignore_errors=True)

    if failures:
        bench_print("\nFailures:")
        for failure in failures:
            bench_print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchError as exc:
        bench_print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
