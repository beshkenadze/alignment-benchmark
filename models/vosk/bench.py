#!/usr/bin/env python3
import argparse
import importlib
import json
import subprocess
import sys
import tempfile
import time
import tracemalloc
import wave
from pathlib import Path
from typing import Protocol, TypedDict, cast


class WordResult(TypedDict):
    word: str
    start: float | None
    end: float | None
    confidence: float | None


class ModelSpec(TypedDict):
    display_name: str
    model_path: Path


class VoskModel(Protocol):
    pass


class VoskModelFactory(Protocol):
    def __call__(self, model_path: str) -> VoskModel: ...


class VoskRecognizer(Protocol):
    def SetWords(self, enable_words: bool) -> None: ...
    def AcceptWaveform(self, data: bytes) -> bool: ...
    def Result(self) -> str: ...
    def FinalResult(self) -> str: ...


class VoskRecognizerFactory(Protocol):
    def __call__(self, model: VoskModel, sample_rate: float) -> VoskRecognizer: ...


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_vosk_factories() -> tuple[VoskModelFactory, VoskRecognizerFactory]:
    vosk_module = importlib.import_module("vosk")
    model_factory = cast(VoskModelFactory, getattr(vosk_module, "Model"))
    recognizer_factory = cast(
        VoskRecognizerFactory, getattr(vosk_module, "KaldiRecognizer")
    )
    return model_factory, recognizer_factory


def require_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")

    raw_map = cast(dict[object, object], value)
    result: dict[str, object] = {}
    for key, item in raw_map.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} has non-string key")
        result[key] = item
    return result


def convert_to_vosk_wav(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    _ = subprocess.run(
        command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def parse_result_words(result_json: dict[str, object]) -> list[WordResult]:
    words: list[WordResult] = []
    raw_items_obj = result_json.get("result")
    if not isinstance(raw_items_obj, list):
        return words

    raw_items = cast(list[object], raw_items_obj)
    for raw_item in raw_items:
        item = require_object(raw_item, "recognizer result entry")

        word_value = item.get("word")
        start_value = item.get("start")
        end_value = item.get("end")
        conf_value = item.get("conf")

        if not isinstance(word_value, str):
            continue

        start_float = (
            float(start_value) if isinstance(start_value, (int, float)) else None
        )
        end_float = float(end_value) if isinstance(end_value, (int, float)) else None
        confidence_float = (
            float(conf_value) if isinstance(conf_value, (int, float)) else None
        )

        words.append(
            {
                "word": word_value,
                "start": start_float,
                "end": end_float,
                "confidence": confidence_float,
            }
        )

    return words


def parse_result_payload(payload: str) -> dict[str, object]:
    parsed = cast(object, json.loads(payload))
    return require_object(parsed, "recognizer JSON payload")


def run_recognition(
    model: VoskModel, recognizer_factory: VoskRecognizerFactory, wav_path: Path
) -> tuple[list[WordResult], float, float]:
    with wave.open(str(wav_path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getframerate() != 16000:
            raise ValueError("Converted audio is not 16kHz mono")

        recognizer = recognizer_factory(model, wav_file.getframerate())
        recognizer.SetWords(True)

        words: list[WordResult] = []
        tracemalloc.start()
        tracemalloc.reset_peak()
        inference_start = time.perf_counter()

        while True:
            data = wav_file.readframes(4000)
            if len(data) == 0:
                break

            if recognizer.AcceptWaveform(data):
                words.extend(
                    parse_result_words(parse_result_payload(recognizer.Result()))
                )

        words.extend(parse_result_words(parse_result_payload(recognizer.FinalResult())))

        inference_time = time.perf_counter() - inference_start
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    return words, inference_time, peak_memory_mb


def resolve_ru_model(script_dir: Path) -> tuple[str, Path]:
    primary_name = "vosk-model-ru-0.42"
    primary_path = script_dir / primary_name
    if primary_path.is_dir():
        return primary_name, primary_path

    fallback_name = "vosk-model-small-ru-0.22"
    fallback_path = script_dir / fallback_name
    if fallback_path.is_dir():
        return fallback_name, fallback_path

    raise FileNotFoundError("No Russian Vosk model found")


def parse_args() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(description="Benchmark Vosk EN/RU timestamps")
    _ = parser.add_argument(
        "--config", required=True, help="Path to benchmark config.json"
    )
    _ = parser.add_argument(
        "--output-dir", required=True, help="Directory for output JSON files"
    )

    parsed = vars(parser.parse_args())
    config_value = parsed.get("config")
    output_dir_value = parsed.get("output_dir")
    if not isinstance(config_value, str):
        raise ValueError("--config must be a string path")
    if not isinstance(output_dir_value, str):
        raise ValueError("--output-dir must be a string path")

    return Path(config_value).resolve(), Path(output_dir_value).resolve()


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    config_path, output_dir = parse_args()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_factory, recognizer_factory = load_vosk_factories()

    with config_path.open("r", encoding="utf-8") as file_handle:
        config = require_object(cast(object, json.load(file_handle)), "config")

    audio_files = require_object(config.get("audio_files"), "config.audio_files")

    ru_model_name, ru_model_path = resolve_ru_model(script_dir)
    model_specs: dict[str, ModelSpec] = {
        "en": {
            "display_name": "vosk (vosk-model-small-en-us-0.15)",
            "model_path": script_dir / "vosk-model-small-en-us-0.15",
        },
        "ru": {
            "display_name": f"vosk ({ru_model_name})",
            "model_path": ru_model_path,
        },
    }

    for lang in ("en", "ru"):
        item = require_object(audio_files.get(lang), f"config.audio_files.{lang}")
        audio_path_value = item.get("path")
        if not isinstance(audio_path_value, str):
            raise ValueError(f"Missing valid path for language {lang}")

        audio_path = Path(audio_path_value).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model_spec = model_specs[lang]
        model_path = model_spec["model_path"]
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        eprint(f"[vosk] loading model for {lang}: {model_path}")
        model_load_start = time.perf_counter()
        model = model_factory(str(model_path))
        model_load_time = time.perf_counter() - model_load_start

        with tempfile.TemporaryDirectory(prefix=f"vosk_{lang}_") as tmpdir:
            converted_wav = Path(tmpdir) / "converted_16k_mono.wav"
            convert_to_vosk_wav(audio_path, converted_wav)

            eprint(f"[vosk] recognizing {lang}: {audio_path}")
            words, inference_time, peak_memory_mb = run_recognition(
                model, recognizer_factory, converted_wav
            )

        payload = {
            "model": model_spec["display_name"],
            "audio_file": str(audio_path),
            "language": lang,
            "inference_time_seconds": round(inference_time, 6),
            "model_load_time_seconds": round(model_load_time, 6),
            "peak_memory_mb": round(peak_memory_mb, 3),
            "words": words,
        }

        output_path = output_dir / f"vosk_{lang}.json"
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)

        eprint(f"[vosk] wrote {output_path} with {len(words)} words")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
