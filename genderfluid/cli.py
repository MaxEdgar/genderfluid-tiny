"""Command line interface for genderfluid-tiny.

Design goals:
- Instant feedback: status/progress goes to stderr and streams live,
  results go to stdout (so piping --json stays clean).
- Nothing silences the user while a model loads or a benchmark runs.
- Help text shows real, copy-pasteable examples.
"""

import argparse
import json
import os
import platform
import sys
import threading
import time

from genderfluid import __version__

SUBCOMMANDS = {"predict", "stats", "benchmark", "interactive"}

# ANSI helpers (only emitted when --color is used and stdout is a TTY)
_C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "red": "\033[31m",
}

_CLASS_COLOR = {
    "girl-associated": _C["magenta"],
    "boy-associated": _C["blue"],
    "uncertain": _C["yellow"],
}

_CONF_COLOR = {"high": _C["green"], "medium": _C["yellow"], "low": _C["dim"]}

_RULE = "=" * 58


def _paint(text: str, color: str) -> str:
    return f"{color}{text}{_C['reset']}"


def _c(text: str, key: str, color: bool) -> str:
    if not color:
        return text
    return _paint(text, _C[key])


# ---------------------------------------------------------------------------
# Progress helpers (stderr only)
# ---------------------------------------------------------------------------

class _Progress:
    """Live progress indicator on stderr. Prints the message instantly, spins
    while work runs, then either reports the elapsed time or clears itself."""

    def __init__(self, message: str):
        self._message = message
        self._start = time.time()
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        sys.stderr.write(self._message + " ")
        sys.stderr.flush()

        def spin():
            frames = "|/-\\"
            i = 0
            while not self._stop.is_set():
                sys.stderr.write(f"\r{self._message} {frames[i % 4]}")
                sys.stderr.flush()
                i += 1
                time.sleep(0.08)

        self._thread = threading.Thread(target=spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
        elapsed = time.time() - self._start
        if exc[0] is None and elapsed >= 0.4:
            sys.stderr.write(f"\r{self._message} done in {elapsed:.1f}s\n")
        else:
            sys.stderr.write("\r" + " " * (len(self._message) + 2) + "\r")
        sys.stderr.flush()
        return False


def _status(message: str):
    return _Progress(message)


def _load_model(model_path=None, label: str = "Loading model"):
    """Load the model with instant user feedback on stderr."""
    from genderfluid import GenderfluidModel

    with _status(label):
        model = GenderfluidModel(model_path)
    return model


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_result(result: dict, color: bool = False) -> str:
    girl = result["girl_associated_probability"] * 100
    boy = result["boy_associated_probability"] * 100
    unc = result["uncertain_probability"] * 100
    classification = result["classification"]
    confidence = result["confidence"]

    lines = [f"Name: {_c(result['name'], 'bold', color)}", ""]

    if color:
        lines.append(f"  girl-associated  {_paint(f'{girl:5.1f}%', _C['magenta'])}")
        lines.append(f"  boy-associated   {_paint(f'{boy:5.1f}%', _C['blue'])}")
        lines.append(f"  uncertain        {unc:5.1f}%")
    else:
        lines.append(f"  girl-associated  {girl:5.1f}%")
        lines.append(f"  boy-associated   {boy:5.1f}%")
        lines.append(f"  uncertain        {unc:5.1f}%")
    lines.append("")

    # Simple 20-cell bar for the leading class
    max_p = max(girl, boy, unc)
    if max_p > 0:
        filled = max(1, round(max_p / 5))
        bar = "#" * filled + "-" * (20 - filled)
        lines.append(f"  [{bar}] {max_p:.1f}% dominant")
        lines.append("")

    cls_label = classification
    if color:
        cls_label = _paint(classification, _CLASS_COLOR.get(classification, _C["bold"]))
    conf_label = confidence
    if color:
        conf_label = _paint(confidence, _CONF_COLOR.get(confidence, _C["reset"]))

    lines.append(f"  Classification: {cls_label}")
    lines.append(f"  Confidence:     {conf_label}")

    if "warning" in result:
        lines.append(f"  Warning: {result['warning']}")
    if "context_warning" in result:
        lines.append(f"  Note: {result['context_warning']}")
    return "\n".join(lines)


def format_compare(results: list[dict], color: bool = False) -> str:
    header = (f"{'Name':<25} {'Classification':<18} {'Girl':>6} {'Boy':>6} "
              f"{'Unc':>6} {'Confidence':<10}")
    lines = [header, "-" * len(header)]
    for r in results:
        g = f"{r['girl_associated_probability']*100:.0f}%"
        b = f"{r['boy_associated_probability']*100:.0f}%"
        u = f"{r['uncertain_probability']*100:.0f}%"
        cls = r["classification"]
        if color:
            cls = _paint(cls, _CLASS_COLOR.get(cls, _C["reset"]))
        conf = r["confidence"]
        if color:
            conf = _paint(conf, _CONF_COLOR.get(conf, _C["reset"]))
        lines.append(f"{r['name']:<25} {cls:<18} {g:>6} {b:>6} {u:>6} {conf:<10}")
    return "\n".join(lines)


def _env_summary() -> str:
    cpu = platform.processor() or platform.machine() or "unknown CPU"
    cores = os.cpu_count() or 0
    return f"{platform.system()} ({cpu}, {cores} cores)"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_predict(args):
    color = args.color and sys.stdout.isatty()
    model = _load_model(args.model)

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
        except OSError as e:
            print(f"Error: cannot read {args.file}: {e}", file=sys.stderr)
            sys.exit(1)
        if not names:
            print("Error: file is empty (expected one name per line)", file=sys.stderr)
            sys.exit(1)

        with _status(f"Classifying {len(names)} names"):
            t0 = time.time()
            results = model.predict_batch(names)
            elapsed = time.time() - t0

        if args.json:
            for r in results:
                print(json.dumps(r, ensure_ascii=False))
        else:
            print(format_compare(results, color=color))
            print(f"\n{len(names)} names in {elapsed*1000:.1f} ms "
                  f"({len(names)/elapsed:,.0f} names/sec)")
        return

    if args.compare:
        names = [n.strip() for n in args.compare if n.strip()]
        if not names:
            print("Error: no names provided", file=sys.stderr)
            sys.exit(1)

        with _status(f"Classifying {len(names)} names"):
            t0 = time.time()
            results = model.predict_batch(names)
            elapsed = time.time() - t0

        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(format_compare(results, color=color))
            print(f"\n{len(names)} names in {elapsed*1000:.1f} ms")
        return

    if args.name is not None:
        name = " ".join(args.name) if isinstance(args.name, list) else args.name
        if not name:
            print("Error: no name given. Example: genderfluid predict Olivia",
                  file=sys.stderr)
            sys.exit(1)
        with _status("Classifying"):
            t0 = time.time()
            result = model.predict(name)
            elapsed = (time.time() - t0) * 1000

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_result(result, color=color))
            if args.verbose:
                print(f"  Latency: {elapsed:.1f} ms")
        return

    _print_predict_usage()


def _print_predict_usage():
    print("Usage: genderfluid predict <name> [options]")
    print()
    print("  genderfluid predict Olivia")
    print("  genderfluid predict Alex Johnson            full names, no quotes needed")
    print("  genderfluid predict --compare Emma James Alex Taylor")
    print("  genderfluid predict --file names.txt        one name per line")
    print("  genderfluid predict Olivia --json")


def cmd_stats(args):
    color = args.color and sys.stdout.isatty()
    model = _load_model(args.model, label="Reading model")
    meta = model.metadata
    size = os.path.getsize(model.model_path)

    if args.json:
        data = {
            "model": "genderfluid-tiny",
            "version": meta.get("version", "unknown"),
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 3),
            "path": model.model_path,
        }
        for key in ("feature_dimensions", "data_source", "train_size", "val_size",
                    "test_size", "test_f1", "validation_f1", "validation_accuracy",
                    "test_accuracy", "ngram_range", "training_date"):
            if key in meta:
                data[key] = meta[key]
        print(json.dumps(data, indent=2, default=str))
        return

    name_c = _paint("genderfluid-tiny", _C["bold"]) if color else "genderfluid-tiny"
    lines = [
        "Model Statistics",
        _RULE,
        f"  Model:      {name_c}",
        f"  Version:    {meta.get('version', 'unknown')}",
        f"  Size:       {size / (1024*1024):.2f} MB ({size:,} bytes)",
    ]
    if "feature_dimensions" in meta:
        ngram = meta.get("ngram_range")
        ngram_s = f" n-grams {ngram[0]}-{ngram[1]}" if isinstance(ngram, (list, tuple)) and len(ngram) == 2 else ""
        lines.append(f"  Features:   {meta['feature_dimensions']:,}{ngram_s}")
    if "data_source" in meta:
        lines.append(f"  Data:       {meta['data_source']}")
    if "training_date" in meta:
        lines.append(f"  Trained:    {meta['training_date']}")
    if "train_size" in meta:
        lines.append(f"  Train:      {meta['train_size']:,}")
    if "val_size" in meta:
        lines.append(f"  Validation: {meta['val_size']:,}")
    if "test_size" in meta:
        lines.append(f"  Test:       {meta['test_size']:,}")
    if "test_f1" in meta:
        lines.append(f"  Test F1:    {meta['test_f1']:.3f}")
    if "validation_f1" in meta:
        lines.append(f"  Val F1:     {meta['validation_f1']:.3f}")
    lines.append("  Classes:    girl-associated, boy-associated, uncertain")
    print("\n".join(lines))


def _benchmark_data(args):
    """Run the benchmark. Returns a dict of measured results."""
    model = _load_model(args.model, label="Loading model")
    size = os.path.getsize(model.model_path)

    test_names = ["Olivia", "James", "Alex", "Isabella", "Noah",
                  "Taylor", "Sophia", "Sam", "Jordan", "Chris"]

    single_reps = 3 if args.quick else 10

    times = []
    with _status("Measuring single-name latency"):
        for _ in range(single_reps):
            for name in test_names:
                t0 = time.time()
                model.predict(name)
                times.append((time.time() - t0) * 1000)
    avg_single = sum(times) / len(times)

    batch_sizes = [10, 100] if args.quick else [10, 100, 1000]
    batches = {}
    for batch_size in batch_sizes:
        batch = (test_names * (batch_size // len(test_names) + 1))[:batch_size]
        with _status(f"Measuring batch of {batch_size} names"):
            t0 = time.time()
            model.predict_batch(batch)
            elapsed = (time.time() - t0) * 1000
        batches[batch_size] = {
            "ms": elapsed,
            "names_per_sec": batch_size / (elapsed / 1000),
        }

    peak_rss = None
    try:
        import resource
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except (ImportError, AttributeError):
        pass

    return {
        "model_size_bytes": size,
        "model_size_mb": size / (1024 * 1024),
        "avg_single_ms": avg_single,
        "batches": batches,
        "peak_rss_mb": peak_rss,
        "env": _env_summary(),
    }


def cmd_benchmark(args):
    color = args.color and sys.stdout.isatty()
    data = _benchmark_data(args)

    if args.json:
        print(json.dumps(data, indent=2))
        return

    title = _paint("Benchmark", _C["bold"]) if color else "Benchmark"
    lines = [title, _RULE]
    lines.append(f"  Model size:      {data['model_size_mb']:.2f} MB")
    lines.append(f"  Single name:     {data['avg_single_ms']:.2f} ms")
    for size, b in data["batches"].items():
        lines.append(f"  Batch {size:>4}:       {b['ms']:>8.1f} ms  "
                     f"({b['names_per_sec']:,.0f} names/sec)")
    if data["peak_rss_mb"] is not None:
        lines.append(f"  Peak RSS:        {data['peak_rss_mb']:.0f} MB")
    lines.append(f"  Environment:     {data['env']}")
    print("\n".join(lines))


def interactive_mode(model=None, color: bool = False):
    print("Name Gender Association Predictor")
    print("Type 'quit' to exit.\n")

    if model is None:
        model = _load_model(None)

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
            result = model.predict(name)
            elapsed = (time.time() - t0) * 1000
            print()
            print(format_result(result, color=color and sys.stdout.isatty()))
            print(f"  ({elapsed:.1f} ms)")
            print()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            break
        except Exception as e:
            print(f"Error: {e}")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _format_action(self, action):
        # Keep argparse defaults but let multi-line epilog/descriptions render.
        return super()._format_action(action)


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="genderfluid",
        formatter_class=_HelpFormatter,
        description="genderfluid-tiny: offline name gender association classifier",
        epilog=(
            "examples:\n"
            "  genderfluid predict Olivia\n"
            "  genderfluid predict Alex Johnson\n"
            "  genderfluid predict --compare Emma James Alex Taylor\n"
            "  genderfluid predict --file names.txt\n"
            "  genderfluid benchmark\n"
            "  genderfluid stats\n"
        ),
    )
    parser.add_argument("--version", action="version",
                        version=f"genderfluid-tiny {__version__}")
    parser.add_argument("--color", action="store_true",
                        help="colored output (auto-disabled when piping)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="show extra details such as latency")

    sub = parser.add_subparsers(dest="command", metavar="{predict,stats,benchmark,interactive}")

    p_predict = sub.add_parser(
        "predict", help="predict gender association for a name",
        description=(
            "Predict the statistical association between a name and gendered\n"
            "naming conventions in the training data.\n"
        ),
        epilog=(
            "examples:\n"
            "  genderfluid predict Olivia\n"
            "  genderfluid predict Alex Johnson        words are joined into one name\n"
            "  genderfluid predict --compare Emma James Alex Taylor\n"
            "  genderfluid predict --file names.txt\n"
            "  genderfluid predict Olivia --json\n"
        ),
        formatter_class=_HelpFormatter,
    )
    p_predict.add_argument("name", nargs="*", metavar="name",
                           help="name to classify (all words are joined)")
    p_predict.add_argument("--file", "-f", metavar="FILE",
                           help="classify every line of FILE (one name per line)")
    p_predict.add_argument("--compare", "-c", nargs="+", metavar="NAME",
                           help="classify several names side by side")
    p_predict.add_argument("--json", "-j", action="store_true",
                           help="machine-readable JSON output")
    p_predict.add_argument("--model", "-m", metavar="PATH", help="path to a custom model file")

    p_stats = sub.add_parser(
        "stats", help="show model statistics",
        description="Show what model is installed, its size, and its training metrics.",
        formatter_class=_HelpFormatter,
    )
    p_stats.add_argument("--model", "-m", metavar="PATH", help="path to a custom model file")
    p_stats.add_argument("--json", "-j", action="store_true", help="JSON output")

    p_bench = sub.add_parser(
        "benchmark", help="run inference benchmark",
        description="Measure model load time, single-name latency, and batch throughput.",
        epilog=(
            "examples:\n"
            "  genderfluid benchmark\n"
            "  genderfluid benchmark --quick\n"
            "  genderfluid benchmark --json\n"
        ),
        formatter_class=_HelpFormatter,
    )
    p_bench.add_argument("--model", "-m", metavar="PATH", help="path to a custom model file")
    p_bench.add_argument("--quick", "-q", action="store_true",
                         help="shorter benchmark (skips the 1000-name batch)")
    p_bench.add_argument("--json", "-j", action="store_true", help="JSON output")

    p_inter = sub.add_parser(
        "interactive", help="interactive prediction mode",
        description="Start a REPL that classifies names as you type them.",
        formatter_class=_HelpFormatter,
    )
    p_inter.add_argument("--model", "-m", metavar="PATH", help="path to a custom model file")

    return parser


def main():
    parser = _build_parser()

    # Backward-compatible shortcut: `genderfluid Olivia` == `genderfluid predict Olivia`
    if len(sys.argv) > 1:
        first = sys.argv[1]
        if first not in SUBCOMMANDS and not first.startswith("-") and first != "--help":
            sys.argv = [sys.argv[0], "predict"] + sys.argv[1:]

    args = parser.parse_args()

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "interactive":
        color = getattr(args, "color", False)
        interactive_mode(_load_model(getattr(args, "model", None)), color=color)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
