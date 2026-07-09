#!/usr/bin/env python3
"""Generate the target-specific, hash-checked non-ROCm dependency lock."""

from __future__ import annotations

import argparse
import email
import hashlib
import os
import platform
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

UV_VERSION = "0.9.22"
PIP_VERSION = "26.1.2"
TARGET_PYTHON = (3, 12)
TARGET_MACHINE = {"amd64", "x86_64"}
TARGET_GLIBC_MINOR = 39
TARGET_PLATFORMS = (
    *(f"manylinux_2_{minor}_x86_64" for minor in range(TARGET_GLIBC_MINOR, 4, -1)),
    "manylinux2014_x86_64",
    "manylinux2010_x86_64",
    "manylinux1_x86_64",
)
HASH_LINE = re.compile(r"--hash=sha256:[0-9a-f]{64}$")
COMPILE_COMMAND = (
    "uv run --isolated --python 3.12 --with-requirements requirements-dev.txt "
    "python scripts/lock_dependencies.py"
)


def canonicalize_name(name: str) -> str:
    """Return the normalized Python package name used for comparisons."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_exact_requirements(path: Path) -> list[tuple[str, str]]:
    """Read a simple file of exact ``name==version`` requirements."""
    requirements: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--") or line.count("==") != 1:
            raise ValueError(f"{path}:{line_number}: expected an exact name==version pin")
        name, version = line.split("==", maxsplit=1)
        requirements.append((canonicalize_name(name), version))
    return requirements


def parse_excluded_names(path: Path) -> set[str]:
    """Read package names excluded because the AMD base owns them."""
    names = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            names.add(canonicalize_name(line))
    return names


def parse_hashed_lock(path: Path) -> list[tuple[str, str]]:
    """Parse and validate the generated one-hash-per-requirement lock format."""
    lines = path.read_text().splitlines()
    requirements: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        if not line.endswith("\\") or line[:-1].strip().count("==") != 1:
            raise ValueError(f"{path}:{index}: expected a continued name==version pin")
        name, version = line[:-1].strip().split("==", maxsplit=1)
        if index >= len(lines) or not HASH_LINE.fullmatch(lines[index].strip()):
            raise ValueError(f"{path}:{index + 1}: expected exactly one SHA-256 hash")
        index += 1
        requirements.append((canonicalize_name(name), version))
    if len(requirements) != len(set(requirements)):
        raise ValueError(f"{path}: duplicate locked requirements")
    return requirements


def wheel_identity(path: Path) -> tuple[str, str]:
    """Read canonical package name and version from a wheel's metadata."""
    with zipfile.ZipFile(path) as wheel:
        metadata_names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError(f"{path}: expected exactly one .dist-info/METADATA file")
        message = email.message_from_bytes(wheel.read(metadata_names[0]))
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise ValueError(f"{path}: wheel metadata has no Name or Version")
    return canonicalize_name(name), version


def sha256(path: Path) -> str:
    """Hash a file without loading the full wheel into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_lock(requirements: list[tuple[str, str]], wheel_paths: list[Path]) -> str:
    """Render one exact target-wheel hash for every resolved requirement."""
    wheels: dict[tuple[str, str], Path] = {}
    for wheel_path in wheel_paths:
        identity = wheel_identity(wheel_path)
        if identity in wheels:
            raise ValueError(f"multiple wheels downloaded for {identity[0]}=={identity[1]}")
        wheels[identity] = wheel_path

    requirement_set = set(requirements)
    missing = requirement_set - wheels.keys()
    unexpected = wheels.keys() - requirement_set
    if missing or unexpected:
        raise ValueError(
            "wheel set does not match resolution: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    lines = [
        "# This file is generated; do not edit it by hand.",
        f"# Regenerate with: {COMPILE_COMMAND}",
        "# Target: CPython 3.12, Linux x86_64, glibc <=2.39, binary wheels only.",
        "# ROCm framework packages are supplied by the digest-pinned AMD base.",
        "",
    ]
    for name, version in requirements:
        lines.append(f"{name}=={version} \\")
        lines.append(f"    --hash=sha256:{sha256(wheels[(name, version)])}")
    return "\n".join(lines) + "\n"


def verify_lock(root: Path, lock_path: Path) -> None:
    """Check lock structure and its relationship to source pins without networking."""
    locked = set(parse_hashed_lock(lock_path))
    direct = set(parse_exact_requirements(root / "requirements.txt"))
    constraints = set(parse_exact_requirements(root / "constraints-pytorch.txt"))
    excluded = parse_excluded_names(root / "requirements-rocm-base.txt")

    missing_direct = direct - locked
    unconstrained = locked - constraints
    forbidden = {name for name, _version in locked} & excluded
    if missing_direct:
        raise ValueError(f"lock is missing direct requirements: {sorted(missing_direct)}")
    if unconstrained:
        raise ValueError(
            f"lock contains requirements absent from constraints: {sorted(unconstrained)}"
        )
    if forbidden:
        raise ValueError(f"lock contains ROCm-owned packages: {sorted(forbidden)}")


def check_environment() -> None:
    """Refuse to generate a target lock from a different interpreter/platform."""
    if sys.version_info[:2] != TARGET_PYTHON:
        raise RuntimeError(f"lock generation requires Python {TARGET_PYTHON[0]}.{TARGET_PYTHON[1]}")
    if sys.platform != "linux" or platform.machine().lower() not in TARGET_MACHINE:
        raise RuntimeError("lock generation requires Linux x86_64")

    uv_version = subprocess.run(
        ["uv", "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if uv_version != f"uv {UV_VERSION}":
        raise RuntimeError(f"expected uv {UV_VERSION}, found {uv_version!r}")

    import pip

    if pip.__version__ != PIP_VERSION:
        raise RuntimeError(f"expected pip {PIP_VERSION}, found {pip.__version__}")


def run(command: list[str], *, cwd: Path) -> None:
    """Run a lock-generation subprocess with visible output."""
    subprocess.run(command, cwd=cwd, check=True)


def generate(root: Path, output: Path) -> None:
    """Resolve versions, download target wheels, and atomically write the lock."""
    constraints = root / "constraints-pytorch.txt"
    direct = root / "requirements.txt"
    excluded = root / "requirements-rocm-base.txt"

    with tempfile.TemporaryDirectory(prefix="framework-rocm-lock-") as temporary:
        temporary_path = Path(temporary)
        resolved = temporary_path / "resolved.txt"
        wheels = temporary_path / "wheels"
        wheels.mkdir()

        run(
            [
                "uv",
                "pip",
                "compile",
                str(direct),
                "--constraints",
                str(constraints),
                "--excludes",
                str(excluded),
                "--only-binary",
                ":all:",
                "--python-version",
                "3.12",
                "--python-platform",
                "x86_64-manylinux_2_39",
                "--no-annotate",
                "--no-header",
                "--quiet",
                "--output-file",
                str(resolved),
            ],
            cwd=root,
        )
        platform_arguments = [
            argument
            for target_platform in TARGET_PLATFORMS
            for argument in ("--platform", target_platform)
        ]
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--disable-pip-version-check",
                "--quiet",
                "--requirement",
                str(resolved),
                "--dest",
                str(wheels),
                "--only-binary=:all:",
                "--no-deps",
                "--python-version",
                "3.12",
                "--implementation",
                "cp",
                "--abi",
                "cp312",
                "--abi",
                "abi3",
                "--abi",
                "none",
                *platform_arguments,
            ],
            cwd=root,
        )

        requirements = parse_exact_requirements(resolved)
        excluded_names = parse_excluded_names(excluded)
        forbidden = {name for name, _version in requirements} & excluded_names
        if forbidden:
            raise RuntimeError(f"ROCm-owned packages entered the lock: {sorted(forbidden)}")

        lock = render_lock(requirements, sorted(wheels.glob("*.whl")))
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_output = tempfile.mkstemp(
            prefix=f".{output.name}.", dir=output.parent
        )
        try:
            with os.fdopen(descriptor, "w") as file:
                file.write(lock)
            os.replace(temporary_output, output)
        finally:
            if os.path.exists(temporary_output):
                os.unlink(temporary_output)
        verify_lock(root, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("requirements-pytorch.lock"),
        help="lock file to write (default: requirements-pytorch.lock)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the committed lock without resolving or downloading",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = args.output if args.output.is_absolute() else root / args.output
    if args.check:
        verify_lock(root, output)
        print(f"verified {output}")
    else:
        check_environment()
        generate(root, output)
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
