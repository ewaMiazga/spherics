# spherics

Minimal Python package scaffold for spherical-distribution functions.

## Structure

```text
spherics/
├── pyproject.toml
├── src/
│   └── spherics/
│       ├── __init__.py
│       └── distributions.py
└── tests/
    └── test_distributions.py
```

Put your implementation in `src/spherics/distributions.py` and expand tests in `tests/`.

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
