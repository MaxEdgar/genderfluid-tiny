#!/usr/bin/env python3
"""Prediction CLI and API for genderfluid-tiny."""

import argparse
import json
import os
import sys

from genderfluid.inference import predict_name, predict_names


def format_result(result: dict) -> str:
    """Format a prediction result for terminal display."""
    lines = []
    lines.append(f"Name: {result['name']}")
    lines.append("")
    lines.append(f"Girl-associated: {result['girl_associated_probability'] * 100:.1f}%")
    lines.append(f"Boy-associated: {result['boy_associated_probability'] * 100:.1f}%")
    lines.append(f"Uncertain: {result['uncertain_probability'] * 100:.1f}%")
    lines.append("")
    lines.append(f"Classification: {result['classification']}")
    lines.append(f"Confidence: {result['confidence']}")
    if "warning" in result:
        lines.append(f"Warning: {result['warning']}")
    if "context_warning" in result:
        lines.append(f"Note: {result['context_warning']}")
    return "\n".join(lines)


def interactive_mode(model_path=None):
    """Run interactive prediction mode."""
    print("Name Gender Association Predictor")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            name = input("Name > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not name or name.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            result = predict_name(name, model_path=model_path)
            print()
            print(format_result(result))
            print()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Genderfluid-tiny: Name Gender Association Predictor"
    )
    parser.add_argument("name", nargs="?", help="Name to predict")
    parser.add_argument("--file", "-f", help="File with names (one per line)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--json", "-j", action="store_true", help="Output JSON")
    parser.add_argument("--model", "-m", help="Path to model file")
    parser.add_argument("--country", help="Country context")
    parser.add_argument("--language", help="Language context")
    parser.add_argument("--info", action="store_true", help="Show model info")

    args = parser.parse_args()

    if args.info:
        model_path = args.model
        if not model_path:
            model_path = os.path.join(
                os.path.dirname(__file__), "models", "genderfluid-tiny.bin"
            )
        if os.path.exists(model_path):
            from genderfluid.model_io import load_model
            _, _, metadata = load_model(model_path)
            print("Model: genderfluid-tiny")
            print(f"Version: {metadata.get('version', 'unknown')}")
            print(f"Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
            print(f"Features: {metadata.get('feature_dimensions', 'unknown')}")
            print(f"Classes: girl-associated, boy-associated, uncertain")
        else:
            print("No model found.")
        return

    if args.interactive:
        interactive_mode(model_path=args.model)
        return

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        results = predict_names(names, model_path=args.model)
        for result in results:
            print(json.dumps(result, ensure_ascii=False))
        return

    if args.name:
        try:
            result = predict_name(
                args.name,
                model_path=args.model,
                country=args.country,
                language=args.language,
            )
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(format_result(result))
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
