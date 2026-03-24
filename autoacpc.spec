# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for autoacpc standalone binary.

Uses one-folder mode for fast startup (no temp extraction on each launch).
Build with: uv run python build.py  (or: uv run python build.py clean)
Output: dist/autoacpc-<OS>-<arch>.tar.gz (.zip on Windows)
"""

import sys
from PyInstaller.utils.hooks import collect_data_files

# strip corrupts Windows DLLs (python3xx.dll) — only enable on Unix
do_strip = sys.platform != "win32"

templateflow_data = collect_data_files("templateflow")

# Heavy modules pulled in transitively (via dipy/scipy) but never used at runtime.
# dipy.core.geometry is the only dipy module we need; it does not import scipy.
excludes = [
    # Unused dipy subpackages (only dipy.core.geometry is needed)
    "dipy.align",
    "dipy.data",
    "dipy.denoise",
    "dipy.direction",
    "dipy.io",
    "dipy.nn",
    "dipy.reconst",
    "dipy.segment",
    "dipy.sims",
    "dipy.tracking",
    "dipy.viz",
    "dipy.workflows",
    # Unused scipy subpackages
    "scipy.stats",
    "scipy.optimize",
    "scipy.special",
    "scipy.sparse",
    "scipy.spatial",
    "scipy.signal",
    "scipy.interpolate",
    "scipy.integrate",
    "scipy.fft",
    "scipy.fftpack",
    "scipy.io",
    "scipy.cluster",
    "scipy.constants",
    "scipy.datasets",
    "scipy.misc",
    "scipy.odr",
    # Testing / dev tools
    "pytest",
    "IPython",
    "jupyter",
    "notebook",
    "sphinx",
    "docutils",
    "setuptools",
    "pip",
    "wheel",
    "distutils",
    # GUI / display (not needed for CLI)
    "tkinter",
    "matplotlib",
    "PIL",
    "pygments",
    # HDF5 (nibabel can use it, but we only load .nii.gz)
    "h5py",
    "tables",
    # Other unused transitive deps
    "pandas",
    "sympy",
    "numba",
    "llvmlite",
    "skimage",
    "sklearn",
    "cvxpy",
    "tqdm",
]

a = Analysis(
    ["entry.py"],
    pathex=["src"],
    binaries=[],
    datas=templateflow_data,
    hiddenimports=[
        "nibabel",
        "nibabel.nifti1",
        "nibabel.nifti2",
        "nibabel.freesurfer",
        "numpy",
        "SimpleITK",
        "dipy",
        "dipy.core",
        "dipy.core.geometry",
        "click",
        "templateflow",
        "templateflow.api",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="autoacpc",
    debug=False,
    bootloader_ignore_signals=False,
    strip=do_strip,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=do_strip,
    upx=True,
    upx_exclude=[],
    name="autoacpc",
)
