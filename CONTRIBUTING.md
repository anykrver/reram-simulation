# Contributing to Neuro-Edge ReRAM Simulator

## Development Setup

```bash
git clone https://github.com/your-username/neuro-edge-reram-simulator.git
cd neuro-edge-reram-simulator
python -m venv venv
# Activate venv (Windows: venv\Scripts\activate; Unix: source venv/bin/activate)
pip install -r requirements.txt
pip install -e ".[dev]"  # if dev extras exist
pip install -e .
```

## Code Style

- **Python:** PEP 8; use type hints and docstrings (Google or NumPy style) for public APIs.
- **Config:** Prefer YAML in `configs/`; avoid hardcoding sizes/seeds in library code.
- **Logging:** Use `src.utils.logger.get_logger()` for INFO/DEBUG; no print in library code.
- **Reproducibility:** Set `numpy.random.seed` (or RNG from config) in scripts/notebooks.

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src  # with coverage
```

## Submitting Changes

1. Fork the repo and create a branch from `main` (or `master`).
2. Make changes; ensure tests pass and `python src/main.py --config configs/ideal.yaml` runs.
3. Open a pull request with a short description and reference any issue.

## CI

Push to `main`/`master` or open a PR to trigger CI (Python 3.8, 3.10, 3.12; tests + smoke run).
