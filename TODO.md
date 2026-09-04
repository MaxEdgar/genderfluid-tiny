# TODO

## Future goals

- Expand training dataset to 1M names (currently 140,536)
- Improve uncertain-class F1 (0.338) -- the weakest class
- Add more Asian sources: Korea, Taiwan, Singapore, Indonesia (APIs blocked scripted access so far)
- Add CLI test coverage (compare/file/json/exit codes currently untested)
- Push overall accuracy beyond 81.0%

---

# Release Checklist

Steps to follow every time you bump the version.

---

## After bumping version

When changing the version number, update ALL of these files:

1. `pyproject.toml` -- `version = "X.Y.Z"`
2. `genderfluid/__init__.py` -- `__version__ = "X.Y.Z"`
   (cli.py reads `__version__` dynamically -- no manual bump needed there)
3. `config.yaml` -- `model.version: "X.Y.Z"` (baked into model metadata by train_finalize.py)
4. `docs/index.html` -- footer `genderfluid-tiny vX.Y.Z`

---

## Before pushing

1. Run tests: `python -m pytest tests/ -v`
2. Run inference: `python predict.py "Olivia"`
3. Verify model exists: `ls -lh models/genderfluid-tiny.bin`
4. Verify no emoji in repo: `grep -rP '[\x{1F300}-\x{1F9FF}]' . --include='*.md' --include='*.py' --include='*.yml'`

---

## Before creating GitHub release

1. All version numbers match across files
2. Tests pass
3. README is up to date (license, features, benchmark)
4. pyproject.toml metadata is correct (description, URLs, keywords)
5. GitHub Actions workflows reference correct files (no references to deleted files)
6. .gitignore excludes: raw data files, build artifacts, __pycache__, .venv

---

## Creating the release

```bash
git add -A
git commit -m "release: vX.Y.Z"
git tag -a vX.Y.Z -m "vX.Y.Z"
git push && git push --tags
gh release create vX.Y.Z --title "vX.Y.Z" --generate-notes
```

IMPORTANT: the release title must keep the leading `v` ("vX.Y.Z", not
"X.Y.Z"). Releases v1.0.3 and v1.0.4 were published without it and had
to be renamed after the fact.

Wait for GitHub Actions to publish to PyPI (check with `gh run list --workflow=release.yml`).

---

## After release

1. Verify on PyPI: `pip index versions genderfluid-tiny`
2. Test clean install: create venv, `pip install genderfluid-tiny==X.Y.Z`, test CLI and API
3. Update DOCUMENTATION.md with what changed

---

## Files that must stay in sync

| File | What to update |
|------|---------------|
| `pyproject.toml` | version, description, URLs, keywords, license |
| `genderfluid/__init__.py` | `__version__` (cli.py reads it dynamically) |
| `config.yaml` | `model.version` (model metadata) |
| `README.md` | badges, property table, comparison table |
| `COMMERCIAL_LICENSE.md` | contact email |
| `docs/index.html` | footer version + hero/feature stats if they change |
| `.github/workflows/release.yml` | should match current structure |
