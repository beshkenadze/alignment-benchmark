#!/usr/bin/env python3
import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import whisperx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WhisperX alignment benchmark")
    parser.add_argument("--config", required=True, help="Path to benchmark config.json")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for JSON outputs"
    )
    return parser.parse_args()


def flatten_words(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in segments:
        for word_info in segment.get("words", []):
            words.append(
                {
                    "word": word_info.get("word"),
                    "start": word_info.get("start"),
                    "end": word_info.get("end"),
                    "confidence": word_info.get("score"),
                }
            )
    return words


def run_one(audio_path: str, language: str) -> dict[str, Any]:
    print(
        f"[whisperx] start language={language} file={audio_path}",
        file=sys.stderr,
        flush=True,
    )

    tracemalloc.start()
    load_start = time.perf_counter()
    model = whisperx.load_model("base", device="cpu", compute_type="int8")
    model_a, metadata = whisperx.load_align_model(language_code=language, device="cpu")
    model_load_time = time.perf_counter() - load_start
    print(
        f"[whisperx] models loaded language={language} in {model_load_time:.3f}s",
        file=sys.stderr,
        flush=True,
    )

    audio = whisperx.load_audio(audio_path)

    transcribe_start = time.perf_counter()
    transcription = model.transcribe(audio, batch_size=4)
    transcribe_time = time.perf_counter() - transcribe_start
    print(
        f"[whisperx] transcribed language={language} in {transcribe_time:.3f}s",
        file=sys.stderr,
        flush=True,
    )

    align_start = time.perf_counter()
    aligned = whisperx.align(
        transcription["segments"], model_a, metadata, audio, device="cpu"
    )
    align_time = time.perf_counter() - align_start
    print(
        f"[whisperx] aligned language={language} in {align_time:.3f}s",
        file=sys.stderr,
        flush=True,
    )

    current, peak = tracemalloc.get_traced_memory()
    _ = current
    tracemalloc.stop()

    words = flatten_words(aligned.get("segments", []))

    return {
        "model": "whisperx (wav2vec2 align, base)",
        "audio_file": audio_path,
        "language": language,
        "inference_time_seconds": align_time,
        "model_load_time_seconds": model_load_time,
        "peak_memory_mb": peak / (1024 * 1024),
        "words": words,
    }


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    audio_files = config.get("audio_files", {})
    if not audio_files:
        raise RuntimeError("No audio_files found in config")

    for lang in ("en", "ru"):
        if lang not in audio_files:
            print(
                f"[whisperx] skipping missing language={lang}",
                file=sys.stderr,
                flush=True,
            )
            continue

        entry = audio_files[lang]
        audio_path = entry.get("path")
        language = entry.get("language", lang)
        if not audio_path:
            print(
                f"[whisperx] skipping language={lang} with empty path",
                file=sys.stderr,
                flush=True,
            )
            continue

        result = run_one(audio_path=audio_path, language=language)
        output_path = output_dir / f"whisperx_{lang}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[whisperx] wrote {output_path}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[whisperx] error: {exc}", file=sys.stderr, flush=True)
        raise
