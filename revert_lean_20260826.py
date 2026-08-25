"""Revert Opsi A LEAN-always (2026-08-26) ke state sebelum eksekusi (Poin A NO BET only).

Backup ada di cache/revert_lean_20260826/*.bak (dibuat sebelum edit).
Jalankan: .venv\\Scripts\\python.exe revert_lean_20260826.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKUP_DIR = ROOT / "cache" / "revert_lean_20260826"

targets = [
    ("format.py.bak", ROOT / "agents" / "football" / "format.py"),
    ("discord_signal_card_accordion.py.bak", ROOT / "agents" / "football" / "discord_signal_card_accordion.py"),
]

ok = True
for bak_name, dest in targets:
    src = BACKUP_DIR / bak_name
    if not src.exists():
        print(f"[FAIL] backup hilang: {src}")
        ok = False
        continue
    shutil.copy2(src, dest)
    print(f"[OK] restored {dest.relative_to(ROOT)} from {src.relative_to(ROOT)}")

if ok:
    print("\nRevert selesai - LEAN kembali hanya saat NO BET (guard di format.py:1628 aktif).")
else:
    print("\nRevert gagal sebagian - cek backup di cache/revert_lean_20260826/")
