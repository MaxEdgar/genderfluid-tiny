# Contributing to genderfluid-tiny

Thanks for your interest in contributing.

## Development setup

```bash
git clone https://github.com/MaxEdgar/genderfluid-tiny.git
cd genderfluid-tiny
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

## Running tests

```bash
pytest tests/ -v
```

All 39 tests must pass before submitting a PR.

## Code style

- Follow existing code patterns
- Keep imports clean (no unused imports)
- No emoji in code, comments, or documentation
- No AI-slop phrasing in documentation

## Pull requests

1. Fork the repository
2. Create a branch from `main`
3. Make your changes
4. Run `pytest tests/ -v`
5. Submit a pull request

Describe what your PR does and why.

## License

By contributing, you agree that your contributions will be licensed under the
Polyform Noncommercial License (same as the project). If you need your
contributions under a different license, state this in your PR.
