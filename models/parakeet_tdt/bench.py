#!/usr/bin/env python3
import argparse
import importlib
import json
import sys
import time
import tracemalloc
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast


class WordItem(TypedDict):
    word: str
    start: float | None
    end: float | None
    confidence: float | None


class OutputPayload(TypedDict):
    model: str
    audio_file: str
    language: str
    inference_time_seconds: float
    model_load_time_seconds: float
    peak_memory_mb: float
    words: list[WordItem]


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _read_attr(obj: object, key: str) -> object | None:
    if hasattr(obj, key):
        return cast(object, getattr(obj, key))
    if isinstance(obj, Mapping):
        obj_map = cast(Mapping[str, object], obj)
        return obj_map.get(key)
    return None


def _to_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _expect_mapping(obj: object, name: str) -> Mapping[str, object]:
    if not isinstance(obj, Mapping):
        raise TypeError(f"{name} must be an object")
    out: dict[str, object] = {}
    obj_map = cast(Mapping[object, object], obj)
    for key, value in obj_map.items():
        if isinstance(key, str):
            out[key] = value
    return out


def _expect_str(item: Mapping[str, object], key: str, name: str) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{name}.{key} must be a string")
    return value


def extract_words(result: object) -> list[WordItem]:
    """Extract word-level timestamps, merging sub-word tokens into words.
    Parakeet outputs token-level pieces (e.g. ' W', 'ell', ',').
    Tokens starting with a space begin a new word; others continue the current word.
    """
    raw_tokens: list[WordItem] = []
    sentences_obj = _read_attr(result, "sentences")
    if not isinstance(sentences_obj, list):
        return raw_tokens

    for sentence_obj in cast(list[object], sentences_obj):
        tokens_obj = _read_attr(sentence_obj, "tokens")
        if not isinstance(tokens_obj, list):
            continue

        for token_obj in cast(list[object], tokens_obj):
            text_obj = _read_attr(token_obj, "text")
            start_obj = _read_attr(token_obj, "start")
            end_obj = _read_attr(token_obj, "end")
            confidence_obj = _read_attr(token_obj, "confidence")

            raw_tokens.append(
                {
                    "word": str(text_obj) if text_obj is not None else "",
                    "start": _to_float(start_obj),
                    "end": _to_float(end_obj),
                    "confidence": _to_float(confidence_obj),
                }
            )

    # Merge sub-word tokens into words
    # A token starting with a space (or the first token) begins a new word
    words: list[WordItem] = []
    for token in raw_tokens:
        text = token["word"]
        is_new_word = text.startswith(" ") or len(words) == 0

        if is_new_word:
            words.append({
                "word": text.strip(),
                "start": token["start"],
                "end": token["end"],
                "confidence": token["confidence"],
            })
        else:
            # Append to current word
            if words:
                words[-1]["word"] += text
                words[-1]["end"] = token["end"]

    # Filter out empty/punctuation-only words
    return [w for w in words if w["word"].strip()]

def transcribe_with_timing(
    model: object, audio_path: str
) -> tuple[object, float, float]:
    transcribe_fn = _read_attr(model, "transcribe")
    if not callable(transcribe_fn):
        raise TypeError("Loaded model has no callable transcribe(audio_path) method")

    tracemalloc.start()
    start_time = time.perf_counter()
    result = transcribe_fn(audio_path)
    inference_time = time.perf_counter() - start_time
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak_bytes / (1024 * 1024)
    return result, inference_time, peak_memory_mb


def load_model(model_name: str) -> object:
    module = importlib.import_module("parakeet_mlx")
    from_pretrained_obj = _read_attr(module, "from_pretrained")
    if not callable(from_pretrained_obj):
        raise TypeError("parakeet_mlx.from_pretrained is not callable")
    return from_pretrained_obj(model_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Parakeet-TDT-0.6B-v3 with native word timestamps"
    )
    _ = parser.add_argument(
        "--config", type=str, required=True, help="Path to benchmark config.json"
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for benchmark output JSON files",
    )
    args = parser.parse_args()

    try:
        config_path = Path(cast(str, args.config)).resolve()
        output_dir = Path(cast(str, args.output_dir)).resolve()
        _ = output_dir.mkdir(parents=True, exist_ok=True)

        with config_path.open("r", encoding="utf-8") as fh:
            config_obj = cast(object, json.load(fh))

        config = _expect_mapping(config_obj, "config")
        audio_files = _expect_mapping(config.get("audio_files"), "config.audio_files")

        eprint("[parakeet-tdt] loading model='mlx-community/parakeet-tdt-0.6b-v3'")
        load_start = time.perf_counter()
        model = load_model("mlx-community/parakeet-tdt-0.6b-v3")
        model_load_time_seconds = time.perf_counter() - load_start
        eprint(f"[parakeet-tdt] model loaded in {model_load_time_seconds:.3f}s")

        for lang in ("en", "ru"):
            item = _expect_mapping(audio_files.get(lang), f"config.audio_files.{lang}")
            audio_path = _expect_str(item, "path", f"config.audio_files.{lang}")

            language_obj = item.get("language")
            language = language_obj if isinstance(language_obj, str) else lang

            eprint(f"[parakeet-tdt] processing language={language} file={audio_path}")
            result, inference_time_seconds, peak_memory_mb = transcribe_with_timing(
                model, audio_path
            )
            words = extract_words(result)

            payload: OutputPayload = {
                "model": "parakeet-tdt-0.6b-v3 (mlx)",
                "audio_file": audio_path,
                "language": language,
                "inference_time_seconds": round(inference_time_seconds, 6),
                "model_load_time_seconds": round(model_load_time_seconds, 6),
                "peak_memory_mb": round(peak_memory_mb, 3),
                "words": words,
            }

            output_path = output_dir / f"parakeet_tdt_{lang}.json"
            with output_path.open("w", encoding="utf-8") as fh:
                _ = json.dump(payload, fh, ensure_ascii=False, indent=2)

            eprint(f"[parakeet-tdt] wrote {output_path} (words={len(words)})")

        return 0
    except Exception as exc:
        eprint(f"[parakeet-tdt] benchmark failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
