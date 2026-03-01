#!/usr/bin/env python3
# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportAny=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportUnknownArgumentType=false
import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import stable_whisper


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _safe_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_words(result) -> list[dict]:
    words: list[dict] = []
    segments = _safe_get(result, "segments", []) or []

    for segment in segments:
        segment_words = _safe_get(segment, "words", []) or []
        for word_obj in segment_words:
            word = _safe_get(word_obj, "word", "")
            start = _safe_get(word_obj, "start")
            end = _safe_get(word_obj, "end")
            confidence = _safe_get(word_obj, "probability")

            if confidence is None:
                confidence = _safe_get(word_obj, "confidence")

            words.append(
                {
                    "word": str(word),
                    "start": float(start) if start is not None else None,
                    "end": float(end) if end is not None else None,
                    "confidence": float(confidence) if confidence is not None else None,
                }
            )

    return words


def run_for_audio(
    model, audio_path: str, language: str, transcript: str
) -> tuple[dict, float, str, int]:
    tracemalloc.start()
    mode_used = "align"

    inference_start = time.perf_counter()
    try:
        align_result = model.align(audio_path, transcript, language=language)
    except Exception as align_error:
        if language == "ru":
            eprint(
                f"[stable-ts] align failed for RU; falling back to transcribe mode: {align_error}"
            )
            mode_used = "transcribe_fallback"
            align_result = model.transcribe(
                audio_path, language=language, word_timestamps=True
            )
        else:
            tracemalloc.stop()
            raise

    inference_time_seconds = time.perf_counter() - inference_start

    words = extract_words(align_result)

    transcribe_start = time.perf_counter()
    transcribe_result = model.transcribe(
        audio_path, language=language, word_timestamps=True
    )
    transcribe_time_seconds = time.perf_counter() - transcribe_start
    transcribe_words = extract_words(transcribe_result)

    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    eprint(
        f"[stable-ts] {language}: mode={mode_used}, align_words={len(words)}, "
        f"transcribe_words={len(transcribe_words)}, transcribe_time={transcribe_time_seconds:.3f}s"
    )

    payload = {
        "audio_file": audio_path,
        "language": language,
        "inference_time_seconds": round(inference_time_seconds, 6),
        "peak_memory_mb": round(peak_memory_bytes / (1024 * 1024), 3),
        "words": words,
    }
    return payload, transcribe_time_seconds, mode_used, len(transcribe_words)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark stable-ts forced alignment")
    parser.add_argument("--config", required=True, help="Path to benchmark config.json")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for benchmark output JSON files"
    )
    args = parser.parse_args()

    try:
        config_path = Path(args.config).resolve()
        output_dir = Path(args.output_dir).resolve()

        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)

        output_dir.mkdir(parents=True, exist_ok=True)

        eprint("[stable-ts] loading model='base' on CPU")
        model_load_start = time.perf_counter()
        model = stable_whisper.load_model("base", device="cpu")
        model_load_time_seconds = time.perf_counter() - model_load_start
        eprint(f"[stable-ts] model loaded in {model_load_time_seconds:.3f}s")

        audio_files = config.get("audio_files", {})
        for lang in ("en", "ru"):
            if lang not in audio_files:
                raise KeyError(f"Missing config.audio_files.{lang}")

            item = audio_files[lang]
            audio_path = item["path"]
            transcript = item["reference_transcript"]
            language = item.get("language", lang)

            eprint(f"[stable-ts] processing language={language} file={audio_path}")
            base_payload, _, mode_used, _ = run_for_audio(
                model, audio_path, language, transcript
            )

            model_name = "stable-ts (align mode, base)"
            if mode_used != "align":
                model_name = "stable-ts (transcribe fallback, base)"

            out_payload = {
                "model": model_name,
                "audio_file": base_payload["audio_file"],
                "language": base_payload["language"],
                "inference_time_seconds": base_payload["inference_time_seconds"],
                "model_load_time_seconds": round(model_load_time_seconds, 6),
                "peak_memory_mb": base_payload["peak_memory_mb"],
                "words": base_payload["words"],
            }

            output_path = output_dir / f"stable_ts_{lang}.json"
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(out_payload, fh, ensure_ascii=False, indent=2)

            eprint(f"[stable-ts] wrote {output_path}")

        return 0
    except Exception as exc:
        eprint(f"[stable-ts] benchmark failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
