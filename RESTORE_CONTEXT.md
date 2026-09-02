# RESTORE_CONTEXT.md

Complete context document for the genderfluid-tiny project.
Read this to instantly understand everything about this codebase.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## What This Project Is

A tiny offline Python classifier that predicts whether a name is statistically
associated with feminine or masculine naming conventions. It does NOT determine
a person's gender identity. It estimates cultural/linguistic naming patterns
from training data.

Output categories:
- `girl-associated` -- name statistically aligns with feminine conventions
- `boy-associated` -- name statistically aligns with masculine conventions
- `uncertain` -- ambiguous, insufficient data, or genuinely unisex

## GitHub Repository

https://github.com/MaxEdgar/genderfluid-tiny

Author: Max Edgar
License: MIT

## Architecture

Character n-gram features (2-5 grams) with hashing trick, fed into logistic
regression (3 classes), with lightweight sigmoid calibration.

```
Input name
  |
Unicode normalization + lowercase
  |
Character n-gram extraction (2-5 grams, 4096 hashing dimensions)
  |
L2 normalize
  |
Logistic regression (3 classes, scikit-learn)
  |
Sigmoid (Platt) calibration (per-class A, B parameters)
  |
Output: girl-associated / boy-associated / uncertain
```

Key design decisions:
- NO CalibratedClassifierCV -- it causes OOM (triples memory). Replaced with
  lightweight sigmoid calibration (6 floats: A, B per class).
- NO large transformer/LLM -- this is a 49KB linear classifier.
- NO GPU required -- pure CPU, works on old dual-core machines.
- NO internet required -- fully offline after install.

## Model Size

- 49,689 bytes (0.05 MB)
- Wheel size when packaged: 63KB
- Well under the 50MB limit

## Model Performance (on held-out test data)

- Test accuracy: 68.9%
- Test macro F1: 0.629
- Per-class F1: girl=0.844, boy=0.664, uncertain=0.380
- Trained on 102,927 real names from SSA (1880-2020) + Census 2020

## Training Data

- U.S. Social Security Administration baby names (1880-2020): 100,364 unique names
- U.S. Census Bureau 2020 Census first names: 53,616 unique names
- Combined: 102,927 unique names with 50+ occurrences
- Labels: 85%+ statistical association = girl/boy, below = uncertain
- Splits: 80% train (82,341), 10% validation (10,292), 10% test (10,294)
- Raw data files (ssa_alldata.txt, census xlsx) are in data/ but gitignored

## File Structure

```
genderfluid-tiny/
├── genderfluid/                  # Main Python package
│   ├── __init__.py              # Exports: predict_name, predict_names,
│   │                            #   GenderfluidModel, classify_name,
│   │                            #   is_girl_name, is_boy_name, name_probability
│   ├── __main__.py              # python -m genderfluid support
│   ├── cli.py                   # Full CLI with subcommands
│   │                            #   predict, compare, stats, benchmark, interactive
│   ├── preprocessing.py         # Unicode-aware name normalization
│   ├── features.py              # Character n-gram feature extraction (hashing trick)
│   ├── classifier.py            # Logistic regression + sigmoid calibration
│   ├── calibration.py           # Calibration error computation
│   ├── inference.py             # GenderfluidModel class + convenience functions
│   ├── model_io.py              # Binary save/load (format v2)
│   └── models/
│       └── genderfluid-tiny.bin  # THE TRAINED MODEL (49KB)
│
├── data/
│   ├── names.jsonl              # Full training dataset (102K names)
│   ├── train.jsonl              # Train split
│   ├── validation.jsonl         # Validation split
│   ├── test.jsonl               # Test split
│   ├── ssa_alldata.txt          # Raw SSA data (33MB, gitignored)
│   ├── census_2020_firstnames_sex.xlsx  # Raw Census data (gitignored)
│   └── README.md                # Data documentation
│
├── models/
│   └── genderfluid-tiny.bin     # Duplicate of model (for backward compat)
│
├── native/                      # C++ inference (optional, no deps)
│   ├── main.cpp
│   ├── model.cpp
│   └── model.h
│
├── tests/
│   └── test_all.py              # 29 tests, all passing
│
├── train.py                     # Training script
├── predict.py                   # Backward-compatible CLI wrapper
├── evaluate.py                  # Evaluation script
├── benchmark.py                 # Benchmarking script
├── prepare_data.py              # Data splitting
├── process_real_data.py         # SSA + Census data processing
│
├── pyproject.toml               # PyPI package config
├── requirements.txt             # scikit-learn, numpy
├── config.yaml                  # Model/training configuration
├── .gitignore
├── .github/workflows/test.yml   # CI: tests + inference
└── README.md
```

## Python API

Three levels of usage:

### 1. Simple one-liners (easiest for scripts)

```python
from genderfluid import classify_name, is_girl_name, is_boy_name, name_probability

classify_name("Emma")       # "girl-associated"
is_girl_name("Emma")        # True
is_boy_name("James")        # True
name_probability("Emma")    # 0.9731
```

### 2. Full result dict

```python
from genderfluid import predict_name, predict_names

result = predict_name("Alex")
# {"name": "Alex", "girl_associated_probability": 0.2692,
#  "boy_associated_probability": 0.5914, "uncertain_probability": 0.1395,
#  "classification": "uncertain", "confidence": "low"}

results = predict_names(["Emma", "James", "Alex"])
```

### 3. Model instance (for repeated use, cached)

```python
from genderfluid import GenderfluidModel

model = GenderfluidModel()
result = model.predict("Elva Retta")
results = model.predict_batch(["Emma", "James", "Alex"])
```

## CLI Usage

After `pip install genderfluid-tiny`:

```bash
genderfluid predict "Elva Retta"                    # human-readable
genderfluid predict --json "Alex"                   # JSON output
genderfluid predict --compare "Emma" "James" "Alex" # comparison table
genderfluid predict --file names.txt                # batch from file
genderfluid interactive                             # interactive mode
genderfluid stats                                   # model info
genderfluid benchmark                               # performance test
genderfluid "Alex"                                  # shorthand
python -m genderfluid "Alex"                        # module invocation
```

Backward-compatible:
```bash
python predict.py "Elva Retta"
```

## PyPI Publishing

Package name: `genderfluid-tiny`
Wheel: 63KB
Dependencies: scikit-learn>=1.0, numpy>=1.20

To publish:
```bash
cd genderfluid-tiny
pip install build twine
python -m build
twine upload dist/*
```

## Key Technical Details

### Feature Extraction (features.py)
- Extracts character n-grams (2-5 grams) from normalized name
- Uses hashing trick: 4096 fixed-size feature vector
- Sign hashing to reduce collisions
- L2 normalized output

### Classifier (classifier.py)
- scikit-learn LogisticRegression (C=1.0, lbfgs solver, max_iter=1000)
- Always trains with all 3 classes (adds synthetic padding for missing classes)
- Lightweight sigmoid calibration: stores 6 floats (A, B per class)
- Confidence thresholds: high >= 90%, medium >= 70%, low < 70%

### Model I/O (model_io.py)
- Binary format v2: magic bytes "GFT\0", version, config JSON, coef, intercept, priors, calib A/B
- Compact: only stores what inference needs
- Feature extractor is stateless (hashing), only config is saved

### Preprocessing (preprocessing.py)
- Unicode normalization (NFKD)
- Lowercase
- Hyphens and apostrophes replaced with spaces
- Punctuation removed
- Extra whitespace collapsed
- Accented characters preserved

### Inference (inference.py)
- Singleton model cache (loads once, reuses)
- Batch prediction for efficiency
- Empty/invalid names return "uncertain" with 33/33/34 split

## Hardware Requirements

- CPU only (no GPU, no CUDA, no AVX required)
- Works on old dual-core CPUs
- RAM: ~200MB during inference (mostly Python/scikit-learn overhead)
- Model itself: 49KB

## Benchmark (Intel Celeron N4000)

```
Model size:       0.05 MB (49,689 bytes)
Loading time:     0.3 ms
Single name:      0.93 ms
Batch (10):       2.5 ms   (4,065 names/sec)
Batch (100):     18.7 ms   (5,335 names/sec)
Batch (1000):   180.1 ms   (5,551 names/sec)
Peak RSS:        198 MB
```

## What NOT To Do

1. DO NOT use CalibratedClassifierCV -- it triples memory and causes OOM on
   machines with <4GB RAM. The project uses lightweight sigmoid calibration instead.

2. DO NOT add large dependencies -- the whole point is tiny + fast. scikit-learn
   and numpy are the only deps.

3. DO NOT use transformers/LLMs -- this is a linear classifier for a simple task.

4. DO NOT change the feature dimensions (4096) without retraining the model.
   The saved model's coef_ matrix shape depends on this.

5. DO NOT move the model file -- it must be at `genderfluid/models/genderfluid-tiny.bin`
   for pip install to include it via package-data config.

6. DO NOT add emoji to anything -- the project style is zero emoji.

7. DO NOT add AI slop to documentation -- write like a human developer.

8. DO NOT commit to git without checking with the user first.

## Git History

```
3ab9e41 Prepare for PyPI publication
b0748ef Add simple script-friendly API functions
065b397 Polish API, CLI, and clean up codebase
0437a28 Initial commit
```

## Tests

29 tests in tests/test_all.py, all passing. Run with:
```bash
pytest tests/ -v
```

Test coverage:
- Preprocessing: normalization, spaces, hyphens, apostrophes, unicode, empty, punctuation, extract_given_names, get_primary_name
- Features: shape, normalization, batch, different/same names, config roundtrip
- Classifier: train/predict, probabilities sum to 1, valid range, uncertain classification
- Calibration: error range
- Model I/O: save/load roundtrip
- Inference: returns dict, probabilities valid, valid classification, empty name, unicode, multi-word, batch, country context warning

## Common Gotchas

1. The model was trained on 85%+ threshold. Names below that are "uncertain".
   This means genuinely unisex names (Alex, Sam, Taylor) will be uncertain.

2. The model is NOT accurate for names outside US/English naming conventions.
   It was trained on SSA + Census data.

3. The native C++ code is optional -- Python inference is the primary interface.

4. The `models/` directory at project root has a copy of the model for backward
   compatibility, but the canonical copy is at `genderfluid/models/`.

5. config.yaml has training parameters but the actual training is done by
   train.py which reads data/names.jsonl directly.

6. The CI workflow (.github/workflows/test.yml) only runs tests + inference,
   not training (too slow/ memory-heavy for CI).

## Privacy

- All inference runs locally
- No names transmitted to external services
- Logging of names disabled by default (config.yaml: privacy.log_names: false)
- The model is trained on public government data (SSA, Census)

## Limitations (be honest in docs)

- Estimates statistical patterns, not gender identity
- US/English-centric training data
- 68.9% accuracy (reasonable for a 49KB model, not production-grade)
- Uncertain class F1 is weak (0.38) -- genuinely hard to classify
- No cultural/regional context (model ignores country/language params beyond warning)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
