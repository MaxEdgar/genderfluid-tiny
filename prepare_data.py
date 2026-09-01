#!/usr/bin/env python3
"""Data preparation and validation script."""

import json
import os
import random


def validate_dataset(filepath: str) -> dict:
    """Validate a JSONL dataset file."""
    stats = {
        "total": 0, "valid": 0, "invalid_json": 0, "empty_names": 0,
        "invalid_labels": 0, "duplicate_names": 0,
        "girl_associated": 0, "boy_associated": 0, "uncertain": 0,
        "unique_names": set(), "names_seen": set(),
    }

    valid_labels = {"girl-associated", "boy-associated", "uncertain"}

    if not os.path.exists(filepath):
        print(f"Warning: {filepath} does not exist.")
        return stats

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stats["total"] += 1
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                continue

            name = entry.get("name", "").strip()
            label = entry.get("label", "").strip()

            if not name:
                stats["empty_names"] += 1
                continue
            if label not in valid_labels:
                stats["invalid_labels"] += 1
                continue

            stats["valid"] += 1
            if name in stats["names_seen"]:
                stats["duplicate_names"] += 1
            stats["names_seen"].add(name)
            stats["unique_names"].add(name.lower())

            if label == "girl-associated":
                stats["girl_associated"] += 1
            elif label == "boy-associated":
                stats["boy_associated"] += 1
            else:
                stats["uncertain"] += 1

    return stats


def print_stats(stats: dict, source: str = "") -> None:
    """Print dataset statistics."""
    header = f"Dataset statistics ({source})" if source else "Dataset statistics"
    print(header)
    print("-" * len(header))
    print(f"Total examples: {stats['total']}")
    print(f"Valid examples: {stats['valid']}")
    print(f"Girl-associated: {stats['girl_associated']}")
    print(f"Boy-associated: {stats['boy_associated']}")
    print(f"Uncertain: {stats['uncertain']}")
    print(f"Unique names: {len(stats['unique_names'])}")
    print()


def create_synthetic_dataset(output_path: str) -> None:
    """
    Create a tiny synthetic dataset for testing the pipeline.
    SYNTHETIC DATA - NOT real-world evidence.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    girl_names = [
        "Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte",
        "Amelia", "Harper", "Evelyn", "Abigail", "Emily", "Ella", "Elizabeth",
        "Camila", "Luna", "Sofia", "Aria", "Scarlett", "Penelope", "Layla",
        "Chloe", "Victoria", "Madison", "Eleanor", "Grace", "Nora", "Riley",
        "Zoey", "Hannah", "Hazel", "Lily", "Ellie", "Violet", "Aurora",
        "Savannah", "Audrey", "Brooklyn", "Bella", "Claire", "Skylar", "Lucy",
        "Paisley", "Anna", "Caroline", "Nova", "Genesis", "Emilia", "Kennedy",
        "Samantha", "Maya", "Willow", "Kinsley", "Naomi", "Aaliyah", "Elena",
        "Sarah", "Ariana", "Allison", "Gabriella", "Alice", "Madelyn", "Cora",
        "Ruby", "Eva", "Serenity", "Autumn", "Adeline", "Hailey", "Gianna",
        "Quinn", "Natalie", "Aubrey", "Josephine", "Rylee", "Arianna", "Finley",
        "Lillian", "Melanie", "Daniella", "Lydia", "Vivian", "Lauren", "Maria",
        "Jasmine", "Mary", "Iris", "Ivy", "Jade", "Elsie", "Melody",
        "Leah", "Piper", "Rosalie", "Marie", "Willa", "Margaret",
        "Danielle", "Elva", "Retta", "Michelle", "Renatta", "Priya",
        "Yuki", "Mei", "Fatima", "Amara", "Zara",
    ]

    boy_names = [
        "James", "Robert", "John", "Michael", "William", "David", "Richard",
        "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
        "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
        "Kenneth", "Kevin", "Brian", "George", "Timothy", "Ronald", "Edward",
        "Jason", "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric",
        "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon", "Benjamin",
        "Samuel", "Raymond", "Gregory", "Frank", "Alexander", "Patrick", "Jack",
        "Dennis", "Jerry", "Tyler", "Aaron", "Jose", "Adam", "Nathan",
        "Henry", "Zachary", "Douglas", "Peter", "Noah", "Ethan", "Liam",
        "Mason", "Logan", "Lucas", "Oliver", "Aiden", "Max", "Leo",
        "Jackson", "Sebastian", "Mateo", "Owen", "Elijah", "Grayson",
        "Marcus", "Terrence", "Kwame", "Dmitri", "Hiroshi", "Javier",
        "Ahmed", "Raj", "Omar", "Ali", "Chen", "Hans", "Pierre",
    ]

    uncertain_names = [
        "Alex", "Sam", "Taylor", "Jordan", "Chris", "Pat", "Jamie",
        "Casey", "Morgan", "Riley", "Quinn", "Dakota", "Reese", "Skyler",
        "Peyton", "Finley", "Hayden", "Emerson", "Rowan", "Sage", "River",
        "Phoenix", "Robin", "Kai", "Nico", "Avery", "Cameron", "Drew",
        "Jessie", "Leslie", "Marion", "Ren", "Sora",
    ]

    entries = []
    for name in girl_names:
        entries.append({"name": name, "label": "girl-associated", "weight": 1.0})
    for name in boy_names:
        entries.append({"name": name, "label": "boy-associated", "weight": 1.0})
    for name in uncertain_names:
        entries.append({"name": name, "label": "uncertain", "weight": 1.0})

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Created synthetic dataset with {len(entries)} entries at {output_path}")
    print("  SYNTHETIC DATA - NOT real-world evidence\n")


def split_dataset(
    filepath: str, output_dir: str,
    train_ratio: float = 0.80, val_ratio: float = 0.10, test_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    """Split dataset into train/validation/test."""
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if entry.get("name", "").strip() and entry.get("label", "").strip() in {
                        "girl-associated", "boy-associated", "uncertain"
                    }:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue

    random.seed(seed)
    random.shuffle(entries)

    n = len(entries)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = entries[:n_train]
    val = entries[n_train:n_train + n_val]
    test = entries[n_train + n_val:]

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in [("train.jsonl", train), ("validation.jsonl", val), ("test.jsonl", test)]:
        path = os.path.join(output_dir, split_name)
        with open(path, "w", encoding="utf-8") as f:
            for entry in split_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_data)} examples")
    print()


def main():
    """Main entry point."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "data", "names.jsonl")
    output_dir = os.path.join(base_dir, "data")

    # Create synthetic data if no dataset exists
    if not os.path.exists(input_path):
        print("No dataset found. Creating synthetic dataset for pipeline testing.\n")
        create_synthetic_dataset(input_path)

    # Validate
    print("Validating dataset...")
    stats = validate_dataset(input_path)
    print_stats(stats, "original")

    # Split
    print("Splitting dataset...")
    split_dataset(input_path, output_dir, seed=42)

    # Validate splits
    for split_name in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
        split_path = os.path.join(output_dir, split_name)
        if os.path.exists(split_path):
            split_stats = validate_dataset(split_path)
            print_stats(split_stats, split_name)


if __name__ == "__main__":
    main()
