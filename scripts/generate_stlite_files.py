"""
Generate stlite deployment artifacts: files.json and public/ directory.

Usage:
    python scripts/generate_stlite_files.py

This script reads all source files needed by the stlite in-browser runner
(index.html) and produces:
  - files.json: JSON object mapping relative paths to file contents.
  - public/: A mirror of the source directories consumed by stlite.

Run this before deploying to Vercel (or any static host that serves index.html).
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories whose contents are bundled for stlite
SOURCE_DIRS = ["src", "configs", "dashboard", "experiments"]

# File extensions to include
INCLUDE_EXTS = {".py", ".yaml", ".yml", ".npy", ".toml", ".json", ".txt", ".md"}

# Files / patterns to skip
SKIP_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def should_include(path: Path) -> bool:
    """Return True if the file should be included in the bundle."""
    for part in path.parts:
        if part in SKIP_NAMES:
            return False
    return path.suffix in INCLUDE_EXTS


def collect_files() -> dict[str, str]:
    """Walk source dirs and collect file path -> content mapping."""
    files: dict[str, str] = {}
    for dirname in SOURCE_DIRS:
        src_dir = ROOT / dirname
        if not src_dir.exists():
            print(f"  ⚠ Skipping missing directory: {dirname}")
            continue
        for fpath in sorted(src_dir.rglob("*")):
            if not fpath.is_file():
                continue
            if not should_include(fpath):
                continue
            rel = fpath.relative_to(ROOT).as_posix()
            try:
                content = fpath.read_text(encoding="utf-8")
                files[rel] = content
            except UnicodeDecodeError:
                # Binary files (e.g. .npy) — skip from JSON but copy to public/
                print(f"  ⚠ Binary file skipped from JSON: {rel}")
    return files


def generate_files_json(files: dict[str, str]) -> None:
    """Write files.json to project root."""
    out = ROOT / "files.json"
    out.write_text(json.dumps(files, ensure_ascii=False), encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"  ✓ files.json ({size_kb:.1f} KB, {len(files)} files)")


def generate_public_dir() -> None:
    """Mirror source dirs into public/ for static hosting."""
    pub = ROOT / "public"
    if pub.exists():
        shutil.rmtree(pub)
    for dirname in SOURCE_DIRS:
        src_dir = ROOT / dirname
        if not src_dir.exists():
            continue
        dst_dir = pub / dirname
        shutil.copytree(
            src_dir,
            dst_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"),
        )
    count = sum(1 for _ in pub.rglob("*") if _.is_file())
    print(f"  ✓ public/ ({count} files)")


def main() -> int:
    print("Generating stlite deployment artifacts...")
    files = collect_files()
    generate_files_json(files)
    generate_public_dir()
    print("Done. Ready for deployment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
