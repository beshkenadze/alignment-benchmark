#!/usr/bin/env python3
"""
Alignment Benchmark Comparison Harness

Reads all results from results/*.json and produces:
1. Side-by-side word timestamp comparison table
2. Cross-model agreement analysis
3. Per-word boundary deviation heatmap (text)
4. Summary statistics
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class ModelResult:
    model: str
    language: str
    inference_time: float
    model_load_time: float
    peak_memory_mb: float | None
    words: list[WordTimestamp]


def load_results(results_dir: Path) -> dict[str, list[ModelResult]]:
    """Load all JSON results, grouped by language."""
    by_lang: dict[str, list[ModelResult]] = defaultdict(list)

    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)

        if "words" not in data or "model" not in data:
            continue

        words = [
            WordTimestamp(
                word=w["word"],
                start=w["start"],
                end=w["end"],
                confidence=w.get("confidence"),
            )
            for w in data["words"]
        ]

        result = ModelResult(
            model=data["model"],
            language=data["language"],
            inference_time=data.get("inference_time_seconds", 0),
            model_load_time=data.get("model_load_time_seconds", 0),
            peak_memory_mb=data.get("peak_memory_mb"),
            words=words,
        )
        by_lang[result.language].append(result)

    return dict(by_lang)


def normalize_word(w: str) -> str:
    """Normalize word for matching across models."""
    return w.strip().lower().rstrip(".,!?;:\"'").lstrip("\"'")


def align_word_lists(results: list[ModelResult]) -> list[dict]:
    """
    Align word lists across models using sequential matching.
    Returns list of dicts: {normalized_word, models: {model_name: WordTimestamp}}
    """
    # Use the model with most words as reference
    ref = max(results, key=lambda r: len(r.words))
    aligned = []

    for ref_word in ref.words:
        entry = {
            "word": ref_word.word,
            "normalized": normalize_word(ref_word.word),
            "models": {ref.model: ref_word},
        }

        for other in results:
            if other.model == ref.model:
                continue
            # Find closest matching word by normalized text and position
            best_match = None
            best_dist = float("inf")
            for ow in other.words:
                if normalize_word(ow.word) == entry["normalized"]:
                    dist = abs(ow.start - ref_word.start)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = ow
            if best_match and best_dist < 5.0:  # within 5 seconds
                entry["models"][other.model] = best_match

        aligned.append(entry)

    return aligned


def compute_cross_model_stats(aligned: list[dict], model_names: list[str]) -> None:
    """Compute and print cross-model agreement statistics."""
    print("\n" + "=" * 100)
    print("CROSS-MODEL AGREEMENT ANALYSIS")
    print("=" * 100)

    total_words = len(aligned)
    all_agree_count = 0
    start_devs = defaultdict(list)
    end_devs = defaultdict(list)

    for entry in aligned:
        starts = []
        ends = []
        for model, wt in entry["models"].items():
            starts.append((model, wt.start))
            ends.append((model, wt.end))

        if len(starts) >= 2:
            mean_start = sum(s for _, s in starts) / len(starts)
            mean_end = sum(e for _, e in ends) / len(ends)

            for model, s in starts:
                start_devs[model].append(abs(s - mean_start) * 1000)  # ms
            for model, e in ends:
                end_devs[model].append(abs(e - mean_end) * 1000)  # ms

            max_start_spread = (
                max(s for _, s in starts) - min(s for _, s in starts)
            ) * 1000
            if max_start_spread < 50:  # all within 50ms
                all_agree_count += 1

    print(
        f"\nWords where all models agree (within 50ms): {all_agree_count}/{total_words} "
        f"({100 * all_agree_count / max(total_words, 1):.1f}%)"
    )

    print(
        f"\n{'Model':<25} {'Mean Start Dev (ms)':>20} {'Mean End Dev (ms)':>20} "
        f"{'Median Start (ms)':>20} {'Median End (ms)':>18}"
    )
    print("-" * 105)

    for model in model_names:
        if model in start_devs:
            s = sorted(start_devs[model])
            e = sorted(end_devs[model])
            print(
                f"{model:<25} {sum(s) / len(s):>20.1f} {sum(e) / len(e):>20.1f} "
                f"{s[len(s) // 2]:>20.1f} {e[len(e) // 2]:>18.1f}"
            )


def print_word_comparison(aligned: list[dict], model_names: list[str]) -> None:
    """Print side-by-side word timestamp comparison."""
    print("\n" + "=" * 100)
    print("WORD-BY-WORD TIMESTAMP COMPARISON")
    print("=" * 100)

    # Header
    header = f"{'Word':<20}"
    for name in model_names:
        short = name[:12]
        header += f" | {short:>12} start  {short:>12} end  "
    print(header)
    print("-" * len(header))

    for entry in aligned:
        line = f"{entry['word']:<20}"
        for name in model_names:
            if name in entry["models"]:
                wt = entry["models"][name]
                line += f" | {wt.start:>12.3f}s  {wt.end:>12.3f}s  "
            else:
                line += f" | {'—':>12}   {'—':>12}   "
        print(line)


def print_summary(results: list[ModelResult]) -> None:
    """Print performance summary table."""
    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY")
    print("=" * 100)

    print(
        f"\n{'Model':<25} {'Lang':>4} {'Words':>6} {'Infer Time':>12} "
        f"{'Load Time':>12} {'Peak Mem':>10} {'ms/word':>10}"
    )
    print("-" * 85)

    for r in sorted(results, key=lambda x: (x.language, x.model)):
        mem = f"{r.peak_memory_mb:.0f}MB" if r.peak_memory_mb else "—"
        ms_per_word = (r.inference_time * 1000 / len(r.words)) if r.words else 0
        print(
            f"{r.model:<25} {r.language:>4} {len(r.words):>6} "
            f"{r.inference_time:>11.2f}s {r.model_load_time:>11.2f}s "
            f"{mem:>10} {ms_per_word:>9.1f}"
        )


def print_boundary_heatmap(aligned: list[dict], model_names: list[str]) -> None:
    """Print visual heatmap of start-time deviations across models."""
    print("\n" + "=" * 100)
    print("START-TIME DEVIATION HEATMAP (ms from consensus mean)")
    print("=" * 100)

    header = f"{'Word':<20}"
    for name in model_names:
        header += f" | {name[:15]:>15}"
    print(header)
    print("-" * len(header))

    for entry in aligned:
        starts = {m: wt.start for m, wt in entry["models"].items()}
        if len(starts) < 2:
            continue

        mean_start = sum(starts.values()) / len(starts)
        line = f"{entry['word']:<20}"

        for name in model_names:
            if name in starts:
                dev_ms = (starts[name] - mean_start) * 1000
                # Color-code: ≤25ms green, ≤50ms yellow, >50ms red
                if abs(dev_ms) <= 25:
                    marker = f"{dev_ms:>+8.0f}  ✓"
                elif abs(dev_ms) <= 50:
                    marker = f"{dev_ms:>+8.0f}  ~"
                else:
                    marker = f"{dev_ms:>+8.0f}  ✗"
                line += f" | {marker:>15}"
            else:
                line += f" | {'—':>15}"
        print(line)


def main():
    results_dir = Path(__file__).parent / "results"

    if not results_dir.exists() or not list(results_dir.glob("*.json")):
        print(f"No results found in {results_dir}")
        print("Run individual model benchmarks first.")
        sys.exit(1)

    by_lang = load_results(results_dir)

    for lang, results in sorted(by_lang.items()):
        print(f"\n{'#' * 100}")
        print(f"# LANGUAGE: {lang.upper()}")
        print(f"{'#' * 100}")

        model_names = sorted(set(r.model for r in results))
        print(f"Models: {', '.join(model_names)}")

        print_summary(results)

        aligned = align_word_lists(results)
        print_word_comparison(aligned, model_names)
        print_boundary_heatmap(aligned, model_names)
        compute_cross_model_stats(aligned, model_names)

    # Export comparison as JSON
    export = {}
    for lang, results in by_lang.items():
        aligned = align_word_lists(results)
        export[lang] = {
            "models": [r.model for r in results],
            "performance": [
                {
                    "model": r.model,
                    "inference_time": r.inference_time,
                    "load_time": r.model_load_time,
                    "peak_memory_mb": r.peak_memory_mb,
                    "word_count": len(r.words),
                }
                for r in results
            ],
            "aligned_words": [
                {
                    "word": e["word"],
                    "timestamps": {
                        m: {"start": wt.start, "end": wt.end}
                        for m, wt in e["models"].items()
                    },
                }
                for e in aligned
            ],
        }

    export_path = results_dir / "_comparison.json"
    with open(export_path, "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"\n\nComparison exported to: {export_path}")


if __name__ == "__main__":
    main()
