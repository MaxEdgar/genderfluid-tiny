"""Full-featured CLI for genderfluid-tiny."""

import argparse
import json
import sys
import os
import time

SUBCOMMANDS = {"predict", "stats", "benchmark", "interactive"}


def _load_model(model_path=None):
    from genderfluid import GenderfluidModel
    return GenderfluidModel(model_path)


def format_result(result: dict, color: bool = False) -> str:
    lines = [f"Name: {result['name']}", ""]

    girl = result["girl_associated_probability"] * 100
    boy = result["boy_associated_probability"] * 100
    unc = result["uncertain_probability"] * 100

    if color:
        lines.append(f"Girl-associated: \033[35m{girl:.1f}%\033[0m")
        lines.append(f"Boy-associated:  \033[34m{boy:.1f}%\033[0m")
        lines.append(f"Uncertain:       {unc:.1f}%")
    else:
        lines.append(f"Girl-associated: {girl:.1f}%")
        lines.append(f"Boy-associated:  {boy:.1f}%")
        lines.append(f"Uncertain:       {unc:.1f}%")

    lines.append("")
    lines.append(f"Classification: {result['classification']}")
    lines.append(f"Confidence:     {result['confidence']}")

    if "warning" in result:
        lines.append(f"Warning: {result['warning']}")
    if "context_warning" in result:
        lines.append(f"Note: {result['context_warning']}")
    return "\n".join(lines)


def format_compare(results: list[dict]) -> str:
    lines = [
        f"{'Name':<25} {'Classification':<20} {'Girl':>6} {'Boy':>6} {'Confidence':<10}",
        "-" * 70,
    ]
    for r in results:
        g = f"{r['girl_associated_probability']*100:.0f}%"
        b = f"{r['boy_associated_probability']*100:.0f}%"
        lines.append(f"{r['name']:<25} {r['classification']:<20} {g:>6} {b:>6} {r['confidence']:<10}")
    return "\n".join(lines)


def interactive_mode(model=None, color: bool = False):
    print("Name Gender Association Predictor")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            name = input("Name > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not name or name.lower() in ("quit", "exit", "q"):
            break

        try:
            t0 = time.time()
            if model:
                result = model.predict(name)
            else:
                from genderfluid import predict_name
                result = predict_name(name)
            elapsed = (time.time() - t0) * 1000
            print()
            print(format_result(result, color=color))
            print(f"  ({elapsed:.1f} ms)")
            print()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            break
        except Exception as e:
            print(f"Error: {e}")


def cmd_predict(args):
    model = _load_model(args.model)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]

        t0 = time.time()
        results = model.predict_batch(names)
        elapsed = time.time() - t0

        if args.json:
            for r in results:
                print(json.dumps(r, ensure_ascii=False))
        else:
            print(format_compare(results))

        print(f"\n{len(names)} names in {elapsed*1000:.1f} ms ({len(names)/elapsed:.0f} names/sec)")
        return

    if args.compare:
        names = [n.strip() for n in args.compare if n.strip()]
        if not names:
            print("Error: no names provided")
            return

        t0 = time.time()
        results = model.predict_batch(names)
        elapsed = time.time() - t0

        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(format_compare(results))
            print(f"\n{len(names)} names in {elapsed*1000:.1f} ms")
        return

    if args.name:
        t0 = time.time()
        result = model.predict(args.name)
        elapsed = (time.time() - t0) * 1000

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_result(result, color=args.color))
            if args.verbose:
                print(f"\n  Latency: {elapsed:.1f} ms")
        return

    print("Usage: genderfluid predict <name> [options]")
    print("       genderfluid predict --compare <name1> <name2> ...")
    print("       genderfluid predict --file names.txt")


def cmd_stats(args):
    model = _load_model(args.model)
    meta = model.metadata
    size = os.path.getsize(model.model_path)

    print("Model Statistics")
    print("=" * 40)
    print(f"  Model:      genderfluid-tiny")
    print(f"  Version:    {meta.get('version', 'unknown')}")
    print(f"  Size:       {size / (1024*1024):.2f} MB ({size:,} bytes)")
    if "feature_dimensions" in meta:
        print(f"  Features:   {meta['feature_dimensions']}")
    if "data_source" in meta:
        print(f"  Data:       {meta['data_source']}")
    if "train_size" in meta:
        print(f"  Train:      {meta['train_size']:,}")
    if "val_size" in meta:
        print(f"  Validation: {meta['val_size']:,}")
    if "test_size" in meta:
        print(f"  Test:       {meta['test_size']:,}")
    if "test_f1" in meta:
        print(f"  Test F1:    {meta['test_f1']:.3f}")
    if "validation_f1" in meta:
        print(f"  Val F1:     {meta['validation_f1']:.3f}")
    print(f"  Classes:    girl-associated, boy-associated, uncertain")


def cmd_benchmark(args):
    from genderfluid import GenderfluidModel

    print("Benchmark")
    print("=" * 40)

    t0 = time.time()
    model = GenderfluidModel(args.model)
    load_time = (time.time() - t0) * 1000
    size = os.path.getsize(model.model_path)
    print(f"  Model size:     {size / (1024*1024):.2f} MB")
    print(f"  Loading time:   {load_time:.1f} ms")

    test_names = ["Emma", "James", "Alex", "Michelle Renatta Chan", "Max", "Taylor",
                  "Elva Retta", "Sam", "Jordan", "Chris"]

    # Single name
    times = []
    for _ in range(10):
        for name in test_names:
            t0 = time.time()
            model.predict(name)
            times.append((time.time() - t0) * 1000)
    avg_single = sum(times) / len(times)
    print(f"  Single name:    {avg_single:.2f} ms")

    # Batch
    for batch_size in [10, 100, 1000]:
        batch = (test_names * (batch_size // len(test_names) + 1))[:batch_size]
        t0 = time.time()
        model.predict_batch(batch)
        elapsed = (time.time() - t0) * 1000
        throughput = batch_size / (elapsed / 1000)
        print(f"  Batch {batch_size:>4}:     {elapsed:>7.1f} ms  ({throughput:,.0f} names/sec)")

    try:
        import resource
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"  Peak RSS:       {mem:.0f} MB")
    except (ImportError, AttributeError):
        pass


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="genderfluid",
        description="genderfluid-tiny: name gender association classifier",
    )
    parser.add_argument("--version", action="version", version="genderfluid-tiny 1.0.0")
    parser.add_argument("--color", action="store_true", help="colored output")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose output")

    sub = parser.add_subparsers(dest="command")

    p_predict = sub.add_parser("predict", help="predict gender association for a name")
    p_predict.add_argument("name", nargs="?", help="name to classify")
    p_predict.add_argument("--file", "-f", help="file with one name per line")
    p_predict.add_argument("--compare", "-c", nargs="+", help="compare multiple names")
    p_predict.add_argument("--json", "-j", action="store_true", help="output JSON")
    p_predict.add_argument("--model", "-m", help="path to model file")

    p_stats = sub.add_parser("stats", help="show model statistics")
    p_stats.add_argument("--model", "-m", help="path to model file")

    p_bench = sub.add_parser("benchmark", help="run inference benchmark")
    p_bench.add_argument("--model", "-m", help="path to model file")

    p_inter = sub.add_parser("interactive", help="interactive prediction mode")
    p_inter.add_argument("--model", "-m", help="path to model file")

    return parser


def main():
    parser = _build_parser()

    # Detect backward-compatible calling: predict.py "Elva Retta"
    # sys.argv[0] ends with predict.py and argv[1] is not a subcommand or flag
    if len(sys.argv) > 1:
        first = sys.argv[1]
        if first not in SUBCOMMANDS and not first.startswith("-") and first != "--help":
            # Looks like a bare name -- treat as: predict <name>
            sys.argv = [sys.argv[0], "predict"] + sys.argv[1:]

    args = parser.parse_args()

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "interactive":
        interactive_mode(_load_model(getattr(args, 'model', None)))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
