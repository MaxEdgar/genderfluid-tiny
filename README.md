# genderfluid-tiny

**Tiny offline Python name-gender classifier. Predicts gender associations from names using ML. 0.19 MB model, CPU only, no API needed.**

`pip install genderfluid-tiny`

---

[![PyPI version](https://img.shields.io/pypi/v/genderfluid-tiny.svg)](https://pypi.org/project/genderfluid-tiny/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-Polyform%20NC-gray.svg)](LICENSE)
[![Model size](https://img.shields.io/badge/model-49%20KB-brightgreen.svg)](models/)

---

**[Documentation](https://maxedgar.github.io/genderfluid-tiny/)** | [PyPI](https://pypi.org/project/genderfluid-tiny/) | [GitHub](https://github.com/MaxEdgar/genderfluid-tiny)

---

## What is genderfluid-tiny?

genderfluid-tiny is a lightweight Python library that predicts whether a name is statistically associated with feminine or masculine naming conventions. It uses a character n-gram classifier trained on 102,927 real names from U.S. Social Security Administration data (1880-2020) and Census 2020 records.

Unlike API-based gender detection services, genderfluid-tiny runs entirely offline. No data leaves your machine. No API key required. The entire model is 0.19 MB.

| Property | Value |
|----------|-------|
| Architecture | Character n-gram + logistic regression |
| Model size | 0.19 MB (193 KB) |
| Training data | 102,927 names (SSA + Census) |
| Inference | CPU only, no GPU needed |
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

Girl-associated: 97.5%
Boy-associated:  0.0%
Uncertain:       2.5%

Classification: girl-associated
Confidence:     high
```

## Python API

### Simple one-liners

```python
from genderfluid import classify_name, is_girl_name, is_boy_name, name_probability

classify_name("Emma")       # "girl-associated"
classify_name("James")      # "boy-associated"
classify_name("Alex")       # "uncertain"

is_girl_name("Emma")        # True
is_boy_name("James")        # True

name_probability("Emma")    # 0.9731
```

### Full result dict

```python
from genderfluid import predict_name, predict_names

result = predict_name("Isabella")
# {"name": "Isabella",
#  "girl_associated_probability": 0.8929,
#  "boy_associated_probability": 0.0486,
#  "uncertain_probability": 0.0585,
#  "classification": "girl-associated",
#  "confidence": "medium"}

results = predict_names(["Emma", "James", "Alex"])
for r in results:
    print(f"{r['name']}: {r['classification']}")
```

### Model instance (for repeated use)

```python
from genderfluid import GenderfluidModel

model = GenderfluidModel()  # loads once, cached
model.predict("Olivia")
model.predict_batch(["Emma", "James", "Alex", "Max", "Taylor"])
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
Character n-gram extraction (2-5 grams)
  |
Hashing trick (4096-dim feature vector)
  |
Logistic regression (3 classes)
  |
Sigmoid calibration
  |
Output: girl-associated / boy-associated / uncertain
```

The classifier extracts character-level patterns from names. Names ending in `-a`, `-ia`, `-ine` tend to be feminine. Names ending in `-o`, `-us`, `-er` tend to be masculine. The model learns these patterns from real data rather than hard-coding rules.

## Accuracy

Tested on held-out test data (10,294 names):

| Metric | Value |
|--------|-------|
| Accuracy | 77.7% |
| Macro F1 | 0.713 |
| Girl-associated F1 | 0.876 |
| Boy-associated F1 | 0.796 |
| Uncertain F1 | 0.466 |

The model is trained on U.S./English naming conventions. Accuracy varies by cultural context.

## Benchmark

Measured on a 2-core x86_64 Linux machine. Results vary by hardware.

```
Model size:       0.19 MB
Single name:      1.8 ms
Batch (10):       4.5 ms   (2,207 names/sec)
Batch (100):     61.8 ms   (1,619 names/sec)
```

Run `genderfluid benchmark` on your own hardware.

## Use cases

- **Data pipelines**: Classify gender associations in CSV/spreadsheet data
- **Name validation**: Check if a name follows typical gender patterns
- **Research**: Analyze naming trends across datasets
- **Privacy-sensitive applications**: Process names without sending data to external APIs
- **Offline applications**: Works without internet connectivity
- **Embedded systems**: 0.19 MB model runs on low-resource devices

## Training data

Built from real public data:

1. **U.S. Social Security Administration** baby names (1880-2020): 100,364 unique names
2. **U.S. Census Bureau** 2020 Census first names: 53,616 unique names

Combined: 102,927 names with 50+ occurrences. Names with 85%+ statistical association are labeled `girl-associated` or `boy-associated`. Below that threshold: `uncertain`.

## Training from source

```bash
python process_real_data.py   # download and process SSA + Census data
python prepare_data.py        # validate and split data
python train.py               # train and save model
python evaluate.py            # evaluate on validation/test splits
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
| Model size | 0.19 MB | 600 KB+ | 10 MB+ |
| License | Polyform NC | GPLv3 | -- |
| Last updated | 2026 | 2016 | -- |
| Approach | ML (n-gram + LR) | Lookup table | ML |
| Uncertain category | Yes | Partial | No |
| pip install | Yes | Yes | Yes |
| Offline | Yes | Yes | Yes |

## Limitations

- Works with full names: first, middle, and last
- U.S./English-centric training data
- Name associations vary by culture, language, and generation
- The `uncertain` category exists for genuinely ambiguous names
- Not suitable for high-stakes decisions

## Privacy

All inference runs locally. Names are not transmitted to any external service. Logging of names is disabled by default.

## FAQ

### What data is it trained on?

102,927 names from U.S. Social Security Administration baby names (1880-2020) and Census 2020 records.

### How accurate is it?

77.7% accuracy on held-out test data (macro F1 0.71). Girl-associated names: 88% F1. Boy-associated names: 80% F1. Uncertain/ambiguous names: 47% F1.

### Does it work offline?

Yes. After `pip install genderfluid-tiny`, no internet connection is needed.

### What Python versions are supported?

Python 3.10, 3.11, 3.12, 3.13.

### Can I retrain the model?

Yes. See the Training from source section above. The training pipeline is included.

### Does it work with non-English names?

The model is trained on U.S./English naming data. It may not work well for names from other cultural contexts. The preprocessing preserves Unicode characters, so names with accents and special characters are handled.

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
├── tests/                # 29 tests
├── pyproject.toml        # Package config
└── README.md
```

## License

Polyform Noncommercial License 1.0.0. Free for personal, educational, and noncommercial use.
Commercial use requires a license. See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
