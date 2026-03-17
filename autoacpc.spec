# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for autoacpc standalone binary."""

a = Analysis(
    ["entry.py"],
    pathex=["src"],
    binaries=[],
    datas=[],
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
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="autoacpc",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)
