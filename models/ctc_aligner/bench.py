#!/usr/bin/env python3
import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    load_audio,
    postprocess_results,
    preprocess_text,
)


MODEL_NAME = "ctc-forced-aligner (MMS-300M)"
LANGUAGE_MAP = {
    "en": "eng",
    "ru": "rus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ctc-forced-aligner word timestamps"
    )
    parser.add_argument("--config", required=True, help="Path to benchmark config.json")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for output JSON files"
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_peak_memory_mb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(max_rss) / (1024.0 * 1024.0)
    return float(max_rss) / 1024.0


def to_word_entry(item: dict[str, Any]) -> dict[str, Any]:
    confidence = item.get("confidence", item.get("score"))
    if confidence is not None:
        confidence = float(confidence)

    return {
        "word": str(item.get("text", "")),
        "start": float(item.get("start", 0.0)),
        "end": float(item.get("end", 0.0)),
        "confidence": confidence,
    }


def align_one(
    model: Any,
    tokenizer: Any,
    audio_path: str,
    transcript: str,
    language_iso639_3: str,
) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()

    waveform = load_audio(audio_path, model.dtype, model.device)
    emissions, stride = generate_emissions(model, waveform, batch_size=4)
    tokens_starred, text_starred = preprocess_text(
        transcript,
        romanize=True,
        language=language_iso639_3,
    )
    tokens = [token for token in tokens_starred if token != "<star>"]
    text_tokens = [token for token in text_starred if token != "<star>"]

    segments, scores, blank_token = get_alignments(emissions, tokens, tokenizer)
    spans = get_spans(tokens, segments, blank_token)
    word_timestamps = postprocess_results(text_tokens, spans, stride, scores)

    inference_time_seconds = time.perf_counter() - start
    words = [to_word_entry(item) for item in word_timestamps]
    return words, inference_time_seconds


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(config_path)
    audio_files = config.get("audio_files", {})
    if not isinstance(audio_files, dict) or not audio_files:
        raise ValueError("config.json must contain a non-empty 'audio_files' object")

    model_load_start = time.perf_counter()
    alignment_model, alignment_tokenizer = load_alignment_model(
        "cpu", dtype=torch.float32
    )
    model_load_time_seconds = time.perf_counter() - model_load_start

    for lang_code, item in audio_files.items():
        if not isinstance(item, dict):
            continue

        audio_path = item.get("path")
        transcript = item.get("reference_transcript")
        if not audio_path or not transcript:
            continue

        language_iso639_3 = item.get("language_iso639_3") or LANGUAGE_MAP.get(lang_code)
        if not language_iso639_3:
            raise ValueError(
                f"Unsupported language code without language_iso639_3: {lang_code}"
            )

        words, inference_time_seconds = align_one(
            alignment_model,
            alignment_tokenizer,
            str(audio_path),
            str(transcript),
            str(language_iso639_3),
        )

        result = {
            "model": MODEL_NAME,
            "audio_file": str(audio_path),
            "language": str(item.get("language", lang_code)),
            "inference_time_seconds": float(inference_time_seconds),
            "model_load_time_seconds": float(model_load_time_seconds),
            "peak_memory_mb": float(get_peak_memory_mb()),
            "words": words,
        }

        output_path = output_dir / f"ctc_aligner_{lang_code}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
