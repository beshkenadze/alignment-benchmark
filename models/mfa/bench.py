#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NotRequired, TypedDict, cast


class AudioFileConfig(TypedDict):
    path: str
    reference_transcript: str
    language: NotRequired[str]


class BenchmarkConfig(TypedDict):
    audio_files: dict[str, AudioFileConfig]


class LanguageSettings(TypedDict):
    acoustic_model: str
    dictionary: str
    json_name: str
    model_name: str


class WordEntry(TypedDict):
    word: str
    start: float
    end: float
    confidence: float | None


class ResultPayload(TypedDict):
    model: str
    audio_file: str
    language: str
    inference_time_seconds: float
    model_load_time_seconds: float
    peak_memory_mb: float
    words: list[WordEntry]


class Args(argparse.Namespace):
    config: str = ""
    output_dir: str = ""


LANGUAGE_CONFIG: dict[str, LanguageSettings] = {
    "en": {
        "acoustic_model": "english_mfa",
        "dictionary": "english_mfa",
        "json_name": "mfa_en.json",
        "model_name": "mfa v3 (english_mfa)",
    },
    "ru": {
        "acoustic_model": "russian_mfa",
        "dictionary": "russian_mfa",
        "json_name": "mfa_ru.json",
        "model_name": "mfa v3 (russian_mfa)",
    },
}


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="MFA v3 forced alignment benchmark")
    _ = parser.add_argument(
        "--config", required=True, help="Path to benchmark config.json"
    )
    _ = parser.add_argument(
        "--output-dir", required=True, help="Directory for result JSON files"
    )
    return parser.parse_args(namespace=Args())


def load_config(config_path: Path) -> BenchmarkConfig:
    raw_obj = cast(object, json.loads(config_path.read_text(encoding="utf-8")))
    if not isinstance(raw_obj, dict):
        raise RuntimeError("config.json must be a JSON object")
    raw = cast(dict[str, object], raw_obj)

    raw_audio_files = raw.get("audio_files")
    if not isinstance(raw_audio_files, dict):
        raise RuntimeError("config.json is missing 'audio_files' object")
    audio_obj = cast(dict[str, object], raw_audio_files)

    audio_files: dict[str, AudioFileConfig] = {}
    for lang in ("en", "ru"):
        entry = audio_obj.get(lang)
        if not isinstance(entry, dict):
            raise RuntimeError(f"config.audio_files.{lang} must be an object")
        entry_obj = cast(dict[str, object], entry)

        path_value = entry_obj.get("path")
        transcript_value = entry_obj.get("reference_transcript")
        language_value = entry_obj.get("language")

        if not isinstance(path_value, str) or not path_value.strip():
            raise RuntimeError(
                f"config.audio_files.{lang}.path must be a non-empty string"
            )
        if not isinstance(transcript_value, str) or not transcript_value.strip():
            raise RuntimeError(
                f"config.audio_files.{lang}.reference_transcript must be a non-empty string"
            )

        config_entry: AudioFileConfig = {
            "path": path_value,
            "reference_transcript": transcript_value,
        }
        if isinstance(language_value, str) and language_value.strip():
            config_entry["language"] = language_value

        audio_files[lang] = config_entry

    return {"audio_files": audio_files}


def resolve_mfa_command(script_dir: Path) -> list[str]:
    local_micromamba = script_dir / ".micromamba" / "bin" / "micromamba"
    venv_mfa = script_dir / ".venv" / "bin" / "mfa"

    candidates: list[list[str]] = []
    if shutil.which("conda"):
        candidates.append(["conda", "run", "-n", "mfa_bench", "mfa"])
    if shutil.which("mamba"):
        candidates.append(["mamba", "run", "-n", "mfa_bench", "mfa"])
    if local_micromamba.exists():
        candidates.append([str(local_micromamba), "run", "-n", "mfa_bench", "mfa"])
    if shutil.which("micromamba"):
        candidates.append(["micromamba", "run", "-n", "mfa_bench", "mfa"])
    if venv_mfa.exists():
        candidates.append([str(venv_mfa)])
    if shutil.which("mfa"):
        candidates.append(["mfa"])

    for cmd in candidates:
        try:
            completed = subprocess.run(
                cmd + ["version"], check=True, capture_output=True, text=True
            )
            version_text = (completed.stdout or completed.stderr).strip()
            eprint(f"[mfa-bench] using command: {' '.join(cmd)}")
            if version_text:
                eprint(f"[mfa-bench] version: {version_text}")
            return cmd
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    raise RuntimeError(
        "Unable to locate a working MFA command. Run setup.sh first to install MFA."
    )


def convert_to_wav16k_mono(source_audio: Path, destination_audio: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required but was not found in PATH")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source_audio),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(destination_audio),
    ]
    _ = subprocess.run(cmd, check=True, capture_output=True, text=True)


def append_if_valid(
    words: list[WordEntry], start: float | None, end: float | None, text: str | None
) -> None:
    if start is None or end is None or text is None:
        return
    normalized = text.strip()
    if not normalized:
        return
    words.append(
        {
            "word": normalized,
            "start": float(start),
            "end": float(end),
            "confidence": None,
        }
    )


def parse_textgrid_words(textgrid_path: Path) -> list[WordEntry]:
    lines = textgrid_path.read_text(encoding="utf-8", errors="replace").splitlines()
    words: list[WordEntry] = []

    in_words_tier = False
    current_start: float | None = None
    current_end: float | None = None
    current_text: str | None = None

    for raw_line in lines:
        line = raw_line.strip()

        if line.startswith("item ["):
            append_if_valid(words, current_start, current_end, current_text)
            in_words_tier = False
            current_start = None
            current_end = None
            current_text = None
            continue

        if line.startswith("name = "):
            in_words_tier = line == 'name = "words"'
            current_start = None
            current_end = None
            current_text = None
            continue

        if not in_words_tier:
            continue

        if line.startswith("intervals ["):
            append_if_valid(words, current_start, current_end, current_text)
            current_start = None
            current_end = None
            current_text = None
            continue

        if line.startswith("xmin = "):
            current_start = float(line.split("=", 1)[1].strip())
            continue

        if line.startswith("xmax = "):
            current_end = float(line.split("=", 1)[1].strip())
            continue

        if line.startswith("text = "):
            match = re.match(r'text = "(.*)"$', line)
            extracted = match.group(1) if match else ""
            current_text = extracted.replace('""', '"')

    append_if_valid(words, current_start, current_end, current_text)
    return words


def validate_payload_schema(payload: ResultPayload) -> None:
    required = {
        "model",
        "audio_file",
        "language",
        "inference_time_seconds",
        "model_load_time_seconds",
        "peak_memory_mb",
        "words",
    }

    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Payload missing required keys: {sorted(missing)}")

    if not payload["words"]:
        raise ValueError("words must not be empty")


def run_alignment(
    *,
    mfa_cmd: list[str],
    language: str,
    input_audio: Path,
    transcript: str,
    acoustic_model: str,
    dictionary_model: str,
) -> tuple[list[WordEntry], float]:
    with tempfile.TemporaryDirectory(prefix=f"mfa_bench_{language}_") as temp_dir:
        temp_root = Path(temp_dir)
        corpus_dir = temp_root / "corpus"
        output_dir = temp_root / "aligned"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        corpus_audio = corpus_dir / f"sample_{language}.wav"
        corpus_txt = corpus_dir / f"sample_{language}.txt"

        convert_to_wav16k_mono(input_audio, corpus_audio)
        _ = corpus_txt.write_text(transcript.strip() + "\n", encoding="utf-8")

        align_cmd = mfa_cmd + [
            "align",
            str(corpus_dir),
            dictionary_model,
            acoustic_model,
            str(output_dir),
            "--clean",
        ]

        eprint(f"[mfa-bench] running: {' '.join(align_cmd)}")
        started = time.perf_counter()
        completed = subprocess.run(
            align_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started

        if completed.stdout.strip():
            eprint("[mfa-bench] align stdout captured")
        if completed.stderr.strip():
            eprint("[mfa-bench] align stderr captured")

        textgrids = sorted(output_dir.glob("*.TextGrid"))
        if not textgrids:
            raise RuntimeError(f"No TextGrid produced in {output_dir}")

        words = parse_textgrid_words(textgrids[0])
        if not words:
            raise RuntimeError(
                f"No words parsed from TextGrid {textgrids[0]} for language={language}"
            )

        return words, elapsed


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    mfa_cmd = resolve_mfa_command(script_dir)
    config = load_config(config_path)

    for lang in ("en", "ru"):
        settings = LANGUAGE_CONFIG[lang]
        entry = config["audio_files"][lang]

        input_audio = Path(entry["path"]).expanduser().resolve()
        if not input_audio.exists():
            raise FileNotFoundError(f"Audio file not found: {input_audio}")

        transcript = entry["reference_transcript"]
        words, elapsed = run_alignment(
            mfa_cmd=mfa_cmd,
            language=lang,
            input_audio=input_audio,
            transcript=transcript,
            acoustic_model=settings["acoustic_model"],
            dictionary_model=settings["dictionary"],
        )

        payload: ResultPayload = {
            "model": settings["model_name"],
            "audio_file": str(input_audio),
            "language": lang,
            "inference_time_seconds": float(round(elapsed, 6)),
            "model_load_time_seconds": 0.0,
            "peak_memory_mb": 0.0,
            "words": words,
        }
        validate_payload_schema(payload)

        output_path = output_dir / settings["json_name"]
        _ = output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        eprint(f"[mfa-bench] wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
