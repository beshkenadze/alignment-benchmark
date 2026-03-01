#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Protocol, TypedDict, cast

MODEL_ID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
MODEL_NAME = "qwen3-forced-aligner (0.6B, mlx-audio)"


class AudioItem(TypedDict):
    path: str
    language: str
    reference_transcript: str


class BenchmarkConfig(TypedDict):
    audio_files: dict[str, AudioItem]


class WordEntry(TypedDict):
    word: str
    start: float | None
    end: float | None
    confidence: float | None


class ResultPayload(TypedDict):
    model: str
    audio_file: str
    language: str
    inference_time_seconds: float
    model_load_time_seconds: float
    peak_memory_mb: float
    words: list[WordEntry]


class AlignerModel(Protocol):
    def generate(self, *args: object, **kwargs: object) -> object: ...


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_config(raw: object) -> BenchmarkConfig:
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object")

    audio_files_raw = raw.get("audio_files")
    if not isinstance(audio_files_raw, dict) or not audio_files_raw:
        raise ValueError("config.audio_files must be a non-empty object")

    parsed: dict[str, AudioItem] = {}
    for lang, item in audio_files_raw.items():
        if not isinstance(lang, str):
            continue
        if not isinstance(item, dict):
            raise ValueError(f"audio_files.{lang} must be an object")

        path = item.get("path")
        transcript = item.get("reference_transcript")
        language = item.get("language", lang)

        if not isinstance(path, str) or not path:
            raise ValueError(f"audio_files.{lang}.path must be a non-empty string")
        if not isinstance(transcript, str) or not transcript:
            raise ValueError(
                f"audio_files.{lang}.reference_transcript must be a non-empty string"
            )
        if not isinstance(language, str) or not language:
            raise ValueError(f"audio_files.{lang}.language must be a non-empty string")

        parsed[lang] = {
            "path": path,
            "language": language,
            "reference_transcript": transcript,
        }

    if not parsed:
        raise ValueError("No valid audio items found in config.audio_files")

    return {"audio_files": parsed}


def _import_attr(module_name: str, attr_name: str) -> object:
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _reset_peak_memory() -> None:
    try:
        reset_fn = _import_attr("mlx.core", "reset_peak_memory")
        if callable(reset_fn):
            reset_fn()
    except Exception:
        return


def _get_peak_memory_mb() -> float:
    try:
        get_peak = _import_attr("mlx.core", "get_peak_memory")
        if callable(get_peak):
            peak_value = get_peak()
            peak_bytes = _safe_float(peak_value)
            if peak_bytes is not None:
                return peak_bytes / 1_000_000.0
    except Exception:
        pass

    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _obj_attr(obj: object, name: str) -> object | None:
    return cast(object | None, getattr(obj, name, None))


def _normalize_word(item: object) -> WordEntry | None:
    if isinstance(item, dict):
        word_obj = item.get("word") or item.get("text") or item.get("token")
        start_obj = item.get("start")
        if start_obj is None:
            start_obj = item.get("start_time")
        end_obj = item.get("end")
        if end_obj is None:
            end_obj = item.get("end_time")
        confidence_obj = item.get("confidence")
        if confidence_obj is None:
            confidence_obj = item.get("score")
    else:
        word_obj = _obj_attr(item, "word") or _obj_attr(item, "text")
        start_obj = _obj_attr(item, "start")
        if start_obj is None:
            start_obj = _obj_attr(item, "start_time")
        end_obj = _obj_attr(item, "end")
        if end_obj is None:
            end_obj = _obj_attr(item, "end_time")
        confidence_obj = _obj_attr(item, "confidence")
        if confidence_obj is None:
            confidence_obj = _obj_attr(item, "score")

    if not isinstance(word_obj, str) or not word_obj.strip():
        return None

    return {
        "word": word_obj,
        "start": _safe_float(start_obj),
        "end": _safe_float(end_obj),
        "confidence": _safe_float(confidence_obj),
    }


def _to_word_items(raw_words: object) -> list[object]:
    if isinstance(raw_words, list):
        return raw_words
    if isinstance(raw_words, tuple):
        return list(raw_words)
    if isinstance(raw_words, Iterable) and not isinstance(
        raw_words, (str, bytes, dict)
    ):
        return list(cast(Iterable[object], raw_words))
    return []


def _extract_words(result: object) -> list[WordEntry]:
    raw_words: object | None = None

    if isinstance(result, dict):
        raw_words = result.get("words")
        if raw_words is None:
            raw_words = result.get("segments")
    else:
        raw_words = _obj_attr(result, "words")
        if raw_words is None:
            raw_words = _obj_attr(result, "segments")
        if (
            raw_words is None
            and isinstance(result, Iterable)
            and not isinstance(result, (str, bytes))
        ):
            raw_words = result

    word_items = _to_word_items(raw_words)
    output: list[WordEntry] = []
    for item in word_items:
        normalized = _normalize_word(item)
        if normalized is not None:
            output.append(normalized)
    return output


def _load_aligner() -> tuple[AlignerModel, float]:
    load_model_obj = _import_attr("mlx_audio.stt.utils", "load_model")
    if not callable(load_model_obj):
        raise RuntimeError("mlx_audio.stt.utils.load_model is not callable")

    load_model = cast(Callable[[str], object], load_model_obj)
    start = time.perf_counter()
    model_obj = load_model(MODEL_ID)
    elapsed = time.perf_counter() - start

    model = cast(AlignerModel, model_obj)
    if not callable(getattr(model, "generate", None)):
        raise RuntimeError("Loaded model does not expose generate()")

    return model, elapsed


def _run_alignment(model: AlignerModel, audio: str, text: str, language: str) -> object:
    call_attempts: tuple[Callable[[], object], ...] = (
        lambda: model.generate(
            audio=audio,
            text=text,
            language=language,
            task="forced_alignment",
        ),
        lambda: model.generate(audio=audio, text=text, language=language),
        lambda: model.generate(audio, text=text, language=language),
    )

    last_error: Exception | None = None
    for call in call_attempts:
        try:
            return call()
        except TypeError as exc:
            last_error = exc

    raise RuntimeError(f"Unable to run forced alignment generate(): {last_error}")


def run_benchmark(config_path: Path, output_dir: Path) -> None:
    config = _parse_config(_read_json(config_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    _reset_peak_memory()
    model, model_load_time = _load_aligner()

    for lang, item in config["audio_files"].items():
        start = time.perf_counter()
        result = _run_alignment(
            model=model,
            audio=item["path"],
            text=item["reference_transcript"],
            language=item["language"],
        )
        inference_time = time.perf_counter() - start

        payload: ResultPayload = {
            "model": MODEL_NAME,
            "audio_file": item["path"],
            "language": item["language"],
            "inference_time_seconds": round(inference_time, 6),
            "model_load_time_seconds": round(model_load_time, 6),
            "peak_memory_mb": round(_get_peak_memory_mb(), 3),
            "words": _extract_words(result),
        }

        output_path = output_dir / f"qwen3_fa_{lang}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3 ForcedAligner via mlx-audio",
    )
    _ = parser.add_argument(
        "--config",
        required=True,
        help="Path to benchmark config JSON",
    )
    _ = parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where qwen3_fa_<lang>.json files are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(config_path=Path(args.config), output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
