# genderfluid tiny

Tiny local name-gender association classifier.

Estimates statistical associations between names and gendered naming conventions
in its training data. Does not determine a person's gender identity.

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen.svg)](tests/)
[![Model size](https://img.shields.io/badge/model-0.05%20MB-brightgreen.svg)](models/)
[![License](https://img.shields.io/badge/license-MIT-gray.svg)](LICENSE)

---

## Overview

A small offline classifier that estimates whether a name is statistically
associated with feminine or masculine naming conventions. The classifier
uses character n-gram features and logistic regression. It outputs three
categories: `girl-associated`, `boy-associated`, and `uncertain`.

| Property | Value |
|----------|-------|
| Architecture | Character n-gram + logistic regression |
| Inference | CPU only |
| Internet | Not required |
| GPU | Not required |
| Model size | 0.05 MB (49,689 bytes) |
| Python | 3.10+ |

---

## Installation

```bash
git clone https://github.com/MaxEdgar/genderfluid-tiny.git
cd genderfluid-tiny
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

After installation, the `genderfluid` command is available system-wide.

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

```bash
python predict.py "Elva Retta"
```

Or after `pip install -e .`:

```bash
genderfluid predict "Elva Retta"
```

Output:

```
Name: Elva Retta

Girl-associated: 97.5%
Boy-associated:  1.2%
Uncertain:       1.2%

Classification: girl-associated
Confidence:     high
```

## Python API

There are two ways to use the Python API.

**Option 1: Model instance (recommended)**

```python
from genderfluid import GenderfluidModel

model = GenderfluidModel()  # loads default model

result = model.predict("Elva Retta")
print(result["classification"])  # "girl-associated"

# Batch prediction (more efficient for many names)
results = model.predict_batch(["Emma", "James", "Alex", "Max", "Taylor"])
for r in results:
    print(f"{r['name']}: {r['classification']} ({r['confidence']})")
```

The model is loaded once and cached. Subsequent predictions are fast.

**Option 2: Convenience functions**

```python
from genderfluid import predict_name, predict_names

result = predict_name("Alex")
print(result["classification"])  # "uncertain"

results = predict_names(["Emma", "James"])
for r in results:
    print(f"{r['name']}: {r['classification']}")
```

**Prediction result format:**

```python
{
    "name": "Elva Retta",
    "girl_associated_probability": 0.975,
    "boy_associated_probability": 0.012,
    "uncertain_probability": 0.012,
    "classification": "girl-associated",
    "confidence": "high"
}
```

Confidence levels: `high` (>= 90%), `medium` (>= 70%), `low` (< 70%).

## CLI

After `pip install -e .`, use the `genderfluid` command. Or run directly
with `python -m genderfluid` or `python predict.py`.

**Predict a name:**

```bash
genderfluid predict "Elva Retta"
genderfluid "Alex"                      # shorthand
python predict.py "Elva Retta"         # backward-compatible
```

**Compare multiple names:**

```bash
genderfluid predict --compare "Emma" "James" "Alex" "Max" "Taylor"
```

Output:

```
Name                      Classification         Girl    Boy Confidence
----------------------------------------------------------------------
Emma                      girl-associated         97%     0% high
James                     boy-associated           8%    79% medium
Alex                      uncertain               27%    59% low
Max                       boy-associated           7%    77% medium
Taylor                    uncertain               42%    17% low

5 names in 31.6 ms
```

**Batch from file:**

```bash
genderfluid predict --file names.txt
```

Reads one name per line, outputs JSONL with timing info.

**JSON output:**

```bash
genderfluid predict --json "Michelle Renatta Chan"
```

**Interactive mode:**

```bash
genderfluid interactive
```

**Model statistics:**

```bash
genderfluid stats
```

**Benchmark:**

```bash
genderfluid benchmark
```

## Model

The classifier works as follows:

```
Input name
  |
Unicode normalization + lowercase
  |
Character n-gram extraction (2-5 grams)
  |
Hashing trick (compact fixed-size feature vector)
  |
Logistic regression (3 classes)
  |
Sigmoid calibration
  |
Output: girl-associated / boy-associated / uncertain
```

Trained on 102,927 real names from SSA (1880-2020) and Census 2020 data.
Test accuracy: 70.7%. Macro F1: 0.64.

## Training

```bash
python process_real_data.py   # download and process SSA + Census data
python prepare_data.py        # validate and split data
python train.py               # train and save model
python evaluate.py            # evaluate on validation/test splits
```

The training script loads and validates the dataset, trains a logistic
regression classifier, calibrates probabilities, and saves the model
to `models/genderfluid-tiny.bin`.

## Dataset

JSONL format, one entry per line:

```json
{"name": "Emma", "label": "girl-associated"}
{"name": "James", "label": "boy-associated"}
{"name": "Alex", "label": "uncertain"}
```

Optional fields: `weight`, `country`, `language`, `year`.

The included dataset is built from real public data:

1. U.S. Social Security Administration baby names (1880-2020)
2. U.S. Census Bureau 2020 Census first names

Processed by `process_real_data.py`. Names with 85% or stronger
statistical association are labeled `girl-associated` or `boy-associated`.
Names below that threshold are `uncertain`.

## Benchmark

Measured on Intel Celeron N4000 @ 1.10GHz, Python 3.14:

```
Model size:       0.05 MB (49,689 bytes)
Loading time:     0.3 ms
Single name:      0.93 ms
Batch (10):       2.5 ms   (4,065 names/sec)
Batch (100):     18.7 ms   (5,335 names/sec)
Batch (1000):   180.1 ms   (5,551 names/sec)
Peak RSS:        198 MB
```

Run `genderfluid benchmark` to measure on your own hardware.

## Limitations

The model estimates statistical patterns in its training data. It does not
determine gender identity.

Name associations vary by culture, language, and generation. The classifier
may be wrong. The `uncertain` category exists for ambiguous cases. Training
data can contain bias.

## Privacy

All inference runs locally. Names are not transmitted to any external service.
Logging of names is disabled by default (`config.yaml`).

## Repository

```
genderfluid-tiny/
├── genderfluid/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── classifier.py
│   ├── calibration.py
│   ├── inference.py
│   └── model_io.py
├── data/
│   ├── names.jsonl
│   └── README.md
├── models/
│   └── genderfluid-tiny.bin
├── native/
│   ├── main.cpp
│   ├── model.cpp
│   └── model.h
├── tests/
│   └── test_all.py
├── train.py
├── predict.py
├── evaluate.py
├── benchmark.py
├── prepare_data.py
├── process_real_data.py
├── requirements.txt
├── pyproject.toml
├── config.yaml
├── .gitignore
└── README.md
```

## License

MIT
