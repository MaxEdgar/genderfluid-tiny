# Data

## Format

JSONL format with one entry per line:

```json
{"name": "Emma", "label": "girl-associated", "weight": 1.0}
{"name": "James", "label": "boy-associated", "weight": 1.0}
{"name": "Alex", "label": "uncertain", "weight": 1.0}
```

### Optional metadata:

```json
{
  "name": "Alex",
  "label": "uncertain",
  "country": "US",
  "language": "English",
  "year": 2000,
  "weight": 1.0
}
```

## Labels

- `girl-associated`: Name is statistically associated with feminine naming conventions
- `boy-associated`: Name is statistically associated with masculine naming conventions
- `uncertain`: Name has approximately equal associations

## Running

```bash
python prepare_data.py
```

This will:
1. Create a synthetic dataset if none exists (clearly marked as synthetic)
2. Validate the dataset
3. Split into train/validation/test
