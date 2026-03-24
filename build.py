#!/usr/bin/env python3
"""Build autoacpc standalone binary and create distributable archive.

Usage:
    uv run python build.py          # build + archive
    uv run python build.py clean    # remove build artifacts
"""

import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"
BUNDLE = DIST / "autoacpc"
SYSTEM = platform.system()  # Darwin, Linux, Windows
ARCH = platform.machine()  # arm64, x86_64, AMD64


def clean():
    for d in ("build", "dist"):
        p = ROOT / d
        if p.exists():
            shutil.rmtree(p)
    for egg in ROOT.glob("*.egg-info"):
        shutil.rmtree(egg)
    print("Cleaned build artifacts.")


def build():
    exe = BUNDLE / ("autoacpc.exe" if SYSTEM == "Windows" else "autoacpc")

    # --- PyInstaller ---
    print("==> Running PyInstaller")
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "autoacpc.spec", "--noconfirm"],
        cwd=ROOT,
        check=True,
    )

    # --- macOS ad-hoc signing ---
    if SYSTEM == "Darwin":
        print("==> Ad-hoc signing for macOS")
        subprocess.run(["codesign", "--force", "--sign", "-", str(exe)], check=True)
        libs = list(BUNDLE.rglob("*.so")) + list(BUNDLE.rglob("*.dylib"))
        for lib in libs:
            subprocess.run(
                ["codesign", "--force", "--sign", "-", str(lib)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    # --- Archive ---
    if SYSTEM == "Windows":
        archive = DIST / f"autoacpc-{SYSTEM}-{ARCH}.zip"
        print(f"==> Creating {archive.name}")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in BUNDLE.rglob("*"):
                zf.write(f, f.relative_to(DIST))
    else:
        archive = DIST / f"autoacpc-{SYSTEM}-{ARCH}.tar.gz"
        print(f"==> Creating {archive.name}")
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(BUNDLE, arcname="autoacpc")

    size_mb = archive.stat().st_size / 1024 / 1024
    print(f"==> Done: {archive} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean()
    else:
        build()
