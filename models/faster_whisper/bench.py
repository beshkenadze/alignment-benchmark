#!/usr/bin/env python3
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnusedCallResult=false, reportAny=false, reportExplicitAny=false
import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

from faster_whisper import WhisperModel


class AudioEntry(TypedDict):
    path: str
    language: str


class BenchmarkWord(TypedDict):
    word: str
    start: float
    end: float
    confidence: float


class BenchmarkResult(TypedDict):
    model: str
    audio_file: str
    language: str
    inference_time_seconds: float
    model_load_time_seconds: float
    peak_memory_mb: float
    words: list[BenchmarkWord]


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def benchmark_language(
    model: WhisperModel, audio_path: str, language: str
) -> tuple[list[BenchmarkWord], float]:
    start_time = time.perf_counter()
    segments, _ = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language,
    )

    words: list[BenchmarkWord] = []
    for segment in segments:
        if not segment.words:
            continue
        for word in segment.words:
            words.append(
                {
                    "word": word.word,
                    "start": float(word.start),
                    "end": float(word.end),
                    "confidence": float(word.probability),
                }
            )

    inference_time = time.perf_counter() - start_time
    return words, inference_time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark faster-whisper native word timestamps"
    )
    parser.add_argument(
        "--config", required=True, type=str, help="Path to benchmark config JSON"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to write output JSON files",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        config_raw: dict[str, Any] = json.load(f)

    audio_files = config_raw.get("audio_files", {})

    model_load_start = time.perf_counter()
    model = WhisperModel("base", device="cpu", compute_type="int8")
    model_load_time = time.perf_counter() - model_load_start

    for lang_key in ("en", "ru"):
        if lang_key not in audio_files:
            continue

        audio_entry: AudioEntry = audio_files[lang_key]
        audio_path = audio_entry["path"]
        language = audio_entry.get("language", lang_key)
        if language not in {"en", "ru"}:
            language = lang_key

        words, inference_time = benchmark_language(model, audio_path, language)

        result: BenchmarkResult = {
            "model": "faster-whisper (native timestamps, base)",
            "audio_file": audio_path,
            "language": language,
            "inference_time_seconds": inference_time,
            "model_load_time_seconds": model_load_time,
            "peak_memory_mb": peak_memory_mb(),
            "words": words,
        }

        output_path = output_dir / f"faster_whisper_{language}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
