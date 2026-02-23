"""Pytest configuration: add project root to path and set defaults."""
import sys
from pathlib import Path

# Add project root so "src" is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
