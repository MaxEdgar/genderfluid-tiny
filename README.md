# genderfluid tiny

Tiny local name-gender association classifier.

Estimates statistical associations between names and gendered naming conventions
in its training data. Does not determine a person's gender identity.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen.svg)](tests/)
[![Model size](https://img.shields.io/badge/model-0.05%20MB-brightgreen.svg)](models/)
[![License](https://img.shields.io/badge/license-MIT-gray.svg)](LICENSE)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Overview

A small offline classifier that estimates whether a name is statistically
associated with feminine or masculine naming conventions. The classifier
uses character n-gram features and logistic regression. It outputs three
categories: `girl-associated`, `boy-associated`, and `uncertain`.

```
──────────────────────────────────────────────────
```

| Property | Value |
|----------|-------|
| Architecture | Character n-gram + logistic regression |
| Inference | CPU only |
| Internet | Not required |
| GPU | Not required |
| Model size | 0.05 MB (49,612 bytes) |
| Python | 3.10+ |

```
──────────────────────────────────────────────────
```

## Installation

```
git clone https://github.com/YOUR_USER/genderfluid-tiny.git
cd genderfluid-tiny
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

```
python generate_names.py
python prepare_data.py
python train.py
python predict.py "Elva Retta"
```

Output (varies by training data):

```
Name: Elva Retta

Girl-associated: 94.0%
Boy-associated: 4.9%
Uncertain: 1.1%

Classification: girl-associated
Confidence: high
```

## Python API

```python
from genderfluid import predict_name

result = predict_name("Alex")
print(result["classification"])
```

```python
from genderfluid import predict_names

results = predict_names(["Emma", "James", "Alex"])
for r in results:
    print(f"{r['name']}: {r['classification']}")
```

## CLI

```
python predict.py "Elva Retta"          # human-readable output
python predict.py --json "Alex"         # JSON output
python predict.py --file names.txt      # batch from file
python predict.py --interactive         # interactive mode
python predict.py --info                # model metadata
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
Probability calibration
  |
Output: girl / boy / uncertain
```

Training automatically sweeps feature dimensions (1024, 2048, 4096, 8192)
and selects the smallest configuration that meets the configured quality
target.

## Training

```
python generate_names.py      # generate synthetic dataset (optional)
python prepare_data.py        # validate and split data
python train.py               # train and save model
python evaluate.py            # evaluate on validation/test splits
```

The training script:

1. Loads and validates the dataset
2. Sweeps feature dimensions
3. Selects the smallest model meeting the F1 target
4. Evaluates on the test set
5. Saves to `models/genderfluid-tiny.bin`

## Dataset

JSONL format, one entry per line:

```
{"name": "Emma", "label": "girl-associated"}
{"name": "James", "label": "boy-associated"}
{"name": "Alex", "label": "uncertain"}
```

Optional fields: `weight`, `country`, `language`, `year`.

The included synthetic dataset (`data/names.jsonl`) is for pipeline testing
only. For production use, provide your own dataset in this format.

## Benchmark

Measured on Intel Celeron N4000 @ 1.10GHz, Python 3.14:

```
Model size:       0.05 MB (49,612 bytes)
Loading time:     19.6 ms
Single name:      0.94 ms
Batch (100):      47.0 ms  (2,128 names/sec)
Batch (1000):     411.1 ms (2,433 names/sec)
Memory (RSS):     202 MB (includes Python overhead)
```

Native C++ inference (`native/`) is faster and uses less memory.

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
├── generate_names.py
├── requirements.txt
├── pyproject.toml
├── config.yaml
├── .gitignore
└── README.md
```

## License

MIT
