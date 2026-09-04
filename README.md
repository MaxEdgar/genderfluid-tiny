# genderfluid-tiny

**Tiny offline Python name-gender classifier. Predicts gender associations from names using ML. 6 MB model, CPU only, no API needed.**

`pip install genderfluid-tiny`

---

[![PyPI version](https://img.shields.io/pypi/v/genderfluid-tiny.svg)](https://pypi.org/project/genderfluid-tiny/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-39%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-Polyform%20NC-gray.svg)](LICENSE)
[![Model size](https://img.shields.io/badge/model-6.0%20MB-brightgreen.svg)](models/)

---

**[Documentation](https://maxedgar.github.io/genderfluid-tiny/)** | [PyPI](https://pypi.org/project/genderfluid-tiny/) | [GitHub](https://github.com/MaxEdgar/genderfluid-tiny)

---

## What is genderfluid-tiny?

genderfluid-tiny is a lightweight Python library that predicts whether a name is statistically associated with feminine or masculine naming conventions. It uses a character n-gram classifier trained on 140,547 real names from U.S. Social Security Administration data (1880-2020), U.S. Census 2020, French INSEE first-name statistics (1900-2024), Japanese newborn surveys, and Chinese government name reports - 911 million recorded births in total.

Unlike API-based gender detection services, genderfluid-tiny runs entirely offline. No data leaves your machine. No API key required. The entire model is 6 MB.

| Property | Value |
|----------|-------|
| Architecture | Character n-gram + logistic regression |
| Model size | 6 MB |
| Training data | 140,547 names (SSA + Census + INSEE + Japan + China) |
| Inference | CPU only, numpy only (no sklearn/scipy) |
| Internet | Not required |
| License | Polyform Noncommercial |
| Python | 3.10+ |

## Install

```bash
pip install genderfluid-tiny
```

That's it. The `genderfluid` command and Python API are available immediately.

## Quick start

```bash
genderfluid predict "Emma"
```

```
Name: Emma

  girl-associated  100.0%
  boy-associated     0.0%
  uncertain          0.0%

  Classification: girl-associated
  Confidence:     high
```

## Python API

### Simple one-liners

```python
from genderfluid import classify_name, is_girl_name, is_boy_name, name_probability

classify_name("Emma")       # "girl-associated"
classify_name("James")      # "boy-associated"
classify_name("Robin")      # "uncertain"

is_girl_name("Emma")        # True
is_boy_name("James")        # True

name_probability("Emma")    # 0.9998
```

### Full result dict

```python
from genderfluid import predict_name, predict_names

result = predict_name("Isabella")
# {"name": "Isabella",
#  "girl_associated_probability": 1.0,
#  "boy_associated_probability": 0.0,
#  "uncertain_probability": 0.0,
#  "classification": "girl-associated",
#  "confidence": "high"}

results = predict_names(["Emma", "James", "Robin"])
for r in results:
    print(f"{r['name']}: {r['classification']}")
```

### Model instance (for repeated use)

```python
from genderfluid import GenderfluidModel

model = GenderfluidModel()  # loads once, cached
model.predict("Olivia")
model.predict_batch(["Emma", "James", "Robin", "Max", "Taylor"])
```

## CLI

```bash
genderfluid predict "Olivia"                       # human-readable
genderfluid predict --json "Alex"                   # JSON output
genderfluid predict --compare "Emma" "James" "Alex" # comparison table
genderfluid predict --file names.txt                # batch from file
genderfluid interactive                             # interactive mode
genderfluid stats                                   # model info
genderfluid benchmark                               # performance test
```

## How it works

```
Input name
  |
Unicode normalization + lowercase
  |
Character n-gram extraction (1-6 grams)
  |
Hashing trick (524,288-dim feature vector)
  |
Logistic regression (3 classes)
  |
Sigmoid calibration
  |
Output: girl-associated / boy-associated / uncertain
```

The classifier extracts character-level patterns from names. Names ending in `-a`, `-ia`, `-ine` tend to be feminine. Names ending in `-o`, `-us`, `-er` tend to be masculine. The model learns these patterns from real data rather than hard-coding rules.

## Accuracy

Tested on held-out test data (10,987 names):

| Metric | Value |
|--------|-------|
| Accuracy | 82.5% |
| Macro F1 | 0.699 |
| Girl-associated F1 | 0.902 |
| Boy-associated F1 | 0.834 |
| Uncertain F1 | 0.360 |

Primarily U.S./European training data with growing Asian (Japanese/Chinese) coverage. Accuracy varies by cultural context.

## Benchmark

Measured on a 2-core x86_64 Linux machine. Results vary by hardware.

```
Model size:       6.00 MB
Single name:      3.45 ms
Batch (10):        2.4 ms   (4,199 names/sec)
Batch (100):      18.5 ms   (5,393 names/sec)
Batch (1000):    144.8 ms   (6,906 names/sec)
Peak RSS:        47 MB
Cold start:      ~0.4 s (CLI to first prediction)
```

Run `genderfluid benchmark` on your own hardware.

## Use cases

- **Data pipelines**: Classify gender associations in CSV/spreadsheet data
- **Name validation**: Check if a name follows typical gender patterns
- **Research**: Analyze naming trends across datasets
- **Privacy-sensitive applications**: Process names without sending data to external APIs
- **Offline applications**: Works without internet connectivity
- **Embedded systems**: 6 MB model runs on low-resource devices

## Training data

Built from real public data:

1. **U.S. Social Security Administration** baby names (1880-2020): 100,364 unique names
2. **U.S. Census Bureau** 2020 Census first names: 53,616 unique names
3. **France INSEE** first names (1900-2024): 48,506 unique names
4. **Japan** Meiji Yasuda newborn-name surveys: 2,387 unique kanji names
5. **China** Ministry of Public Security name reports: 49 official given names

Combined: 140,547 unique names (911 million recorded births). Names with 85%+ statistical association are labeled `girl-associated` or `boy-associated`. Below that threshold: `uncertain`.

## Training from source

```bash
python fetch_multinational_data.py  # build dataset from official sources, write splits
python train.py                     # train and save model
python evaluate.py                  # evaluate on validation/test splits
```

**Sweep training on GitHub Actions** (recommended for retraining): the
`Train Model` workflow runs all 18 configurations in parallel on GitHub
runners (one job per config, ~4 minutes total) and a `finalize` job picks
the best config by validation F1, evaluates it on the test split, and
commits the model. Trigger it under Actions > Train Model > Run workflow,
or run the two steps locally:

```bash
python train_config.py 524288 1-6 160.0 lbfgs  # one configuration
python train_finalize.py                      # pick best, evaluate, save
```

## Dataset format

JSONL, one entry per line:

```json
{"name": "Emma", "label": "girl-associated"}
{"name": "James", "label": "boy-associated"}
{"name": "Alex", "label": "uncertain"}
```

Optional fields: `weight`, `country`, `language`, `year`.

## Comparison with alternatives

| Feature | genderfluid-tiny | gender-guesser | chicksexer |
|---------|-----------------|----------------|------------|
| Model size | 6 MB | 600 KB+ | 10 MB+ |
| License | Polyform NC | GPLv3 | -- |
| Last updated | 2026 | 2016 | -- |
| Approach | ML (n-gram + LR) | Lookup table | ML |
| Uncertain category | Yes | Partial | No |
| pip install | Yes | Yes | Yes |
| Offline | Yes | Yes | Yes |

## Limitations

- Works with full names: first, middle, and last
- Primarily U.S./European training data; accuracy varies by cultural context
- Chinese/Japanese coverage is limited (roughly 2,400 names) and still growing
- Name associations vary by culture, language, and generation
- The `uncertain` category exists for genuinely ambiguous names
- Not suitable for high-stakes decisions

## Privacy

All inference runs locally. Names are not transmitted to any external service. Logging of names is disabled by default.

## FAQ

### What data is it trained on?

140,547 unique names from U.S. SSA baby names (1880-2020), U.S. Census 2020, French INSEE first names (1900-2024), Japanese newborn surveys, and Chinese government name reports - 911 million recorded births.

### How accurate is it?

82.5% accuracy on held-out test data (macro F1 0.699). Girl-associated names: 90% F1. Boy-associated names: 83% F1. Uncertain/ambiguous names: 36% F1.

### Does it work offline?

Yes. After `pip install genderfluid-tiny`, no internet connection is needed.

### What Python versions are supported?

Python 3.10, 3.11, 3.12, 3.13.

### Can I retrain the model?

Yes. See the Training from source section above. The training pipeline is included.

### Does it work with non-English names?

The model is trained mostly on U.S./European data with growing Asian coverage. The preprocessing preserves Unicode characters, so names with accents, hanzi, and other scripts are handled - but accuracy outside the trained cultures is lower and often returns `uncertain`.

## Repository structure

```
genderfluid-tiny/
├── genderfluid/          # Python package
│   ├── __init__.py       # Public API
│   ├── cli.py            # Command-line interface
│   ├── inference.py      # GenderfluidModel class
│   ├── classifier.py     # Logistic regression + calibration
│   ├── features.py       # Character n-gram extraction
│   ├── preprocessing.py  # Name normalization
│   └── model_io.py       # Binary save/load
├── data/                 # Training dataset
├── models/               # Trained model
├── native/               # C++ inference (optional)
├── tests/                # 39 tests
├── pyproject.toml        # Package config
└── README.md
```

## License

Polyform Noncommercial License 1.0.0. Free for personal, educational, and noncommercial use.
Commercial use requires a license. See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md). Security: see [SECURITY.md](SECURITY.md).
