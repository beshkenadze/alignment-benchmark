#!/usr/bin/env python3
# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportAny=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnannotatedClassAttribute=false, reportUnusedImport=false, reportGeneralTypeIssues=false, reportArgumentType=false
import argparse
import importlib
import json
import sys
import time
import tracemalloc
from pathlib import Path


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _safe_get(obj: object, key: str, default: object | None = None) -> object | None:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_words_from_stable_result(result: object) -> list[dict[str, object | None]]:
    words: list[dict[str, object | None]] = []
    segments = _safe_get(result, "segments", [])
    if not isinstance(segments, list):
        return words

    for segment in segments:
        segment_words = _safe_get(segment, "words", [])
        if not isinstance(segment_words, list):
            continue
        for word_obj in segment_words:
            word = _safe_get(word_obj, "word", "")
            start = _safe_get(word_obj, "start")
            end = _safe_get(word_obj, "end")
            confidence = _safe_get(word_obj, "probability")
            if confidence is None:
                confidence = _safe_get(word_obj, "confidence")

            if start is None or end is None:
                continue

            words.append(
                {
                    "word": str(word),
                    "start": float(start),
                    "end": float(end),
                    "confidence": float(confidence) if confidence is not None else None,
                }
            )

    return words


class StableTSCharAligner:
    def __init__(self) -> None:
        import stable_whisper

        self.model_name = "whisper-char-align (stable-ts, large-v3)"
        model_load_start = time.perf_counter()
        self.model = stable_whisper.load_model("large-v3", device="cpu")
        self.model_load_time_seconds = time.perf_counter() - model_load_start

    def align(
        self,
        audio_path: str,
        transcript: str,
        language: str,
    ) -> tuple[list[dict[str, object | None]], float, float]:
        tracemalloc.start()
        inference_start = time.perf_counter()
        align_method = getattr(self.model, "align")
        if not callable(align_method):
            raise TypeError("stable-ts model does not expose callable align()")
        result = align_method(audio_path, transcript, language=language, aligner="new")
        inference_time_seconds = time.perf_counter() - inference_start
        words = extract_words_from_stable_result(result)
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return words, inference_time_seconds, peak_memory_bytes / (1024 * 1024)


class WhisperCharAlignmentFallback:
    def __init__(self, base_dir: Path) -> None:
        self.model_name = "whisper-char-align (30stomercury)"
        repo_dir = base_dir / "third_party" / "whisper-char-alignment"
        if not repo_dir.exists():
            raise RuntimeError(
                f"Fallback repository not found at {repo_dir}. Run setup.sh first."
            )

        sys.path.insert(0, str(repo_dir))

        import whisper
        import torch
        from whisper.tokenizer import get_tokenizer

        self.whisper = whisper
        self.torch = torch
        self.get_tokenizer = get_tokenizer
        self.retokenize = importlib.import_module("retokenize")
        self.timing = importlib.import_module("timing")

        self.audio_samples_per_token = whisper.audio.HOP_LENGTH * 2
        model_load_start = time.perf_counter()
        self.model = whisper.load_model("large-v3")
        self.model = self.model.to("cpu")
        self.model_load_time_seconds = time.perf_counter() - model_load_start

    def align(
        self,
        audio_path: str,
        transcript: str,
        language: str,
    ) -> tuple[list[dict[str, object | None]], float, float]:
        tracemalloc.start()
        inference_start = time.perf_counter()

        normalized_text = self.retokenize.remove_punctuation(transcript)
        tokenizer = self.get_tokenizer(self.model.is_multilingual, language=language)
        text_tokens = self.retokenize.encode(
            normalized_text, tokenizer, aligned_unit_type="char"
        )

        raw_audio = self.whisper.load_audio(audio_path)
        audio_samples = len(raw_audio)
        padded_audio = self.whisper.pad_or_trim(raw_audio)
        mel = self.whisper.log_mel_spectrogram(padded_audio, 80).to(self.model.device)

        tokens = self.torch.tensor(
            [
                *tokenizer.sot_sequence,
                tokenizer.no_timestamps,
                *text_tokens,
                tokenizer.eot,
            ]
        ).to(self.model.device)

        max_frames = max(1, audio_samples // self.audio_samples_per_token)
        attn_w, _ = self.timing.get_attentions(
            mel,
            tokens,
            self.model,
            tokenizer,
            max_frames,
            medfilt_width=3,
            qk_scale=1.0,
        )
        words, start_times, end_times, _, _ = self.timing.force_align(
            attn_w,
            text_tokens,
            tokenizer,
            aligned_unit_type="char",
            aggregation="topk",
            topk=10,
        )

        output_words: list[dict[str, object | None]] = []
        word_count = min(len(words), len(start_times), len(end_times))
        for idx in range(word_count):
            text_word = str(words[idx]).strip()
            if not text_word:
                continue
            output_words.append(
                {
                    "word": text_word,
                    "start": float(start_times[idx]),
                    "end": float(end_times[idx]),
                    "confidence": None,
                }
            )

        inference_time_seconds = time.perf_counter() - inference_start
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return output_words, inference_time_seconds, peak_memory_bytes / (1024 * 1024)


def load_config(config_path: Path) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if not isinstance(data, dict):
        raise ValueError("config.json must contain a JSON object")
    return data


def run_benchmark(
    aligner: object,
    audio_path: str,
    language: str,
    transcript: str,
    output_path: Path,
) -> None:
    align_method = getattr(aligner, "align")
    if not callable(align_method):
        raise TypeError("aligner object does not expose callable align()")

    words, inference_time_seconds, peak_memory_mb = align_method(
        audio_path, transcript, language
    )
    payload = {
        "model": str(getattr(aligner, "model_name")),
        "audio_file": audio_path,
        "language": language,
        "inference_time_seconds": round(float(inference_time_seconds), 6),
        "model_load_time_seconds": round(
            float(getattr(aligner, "model_load_time_seconds")), 6
        ),
        "peak_memory_mb": round(float(peak_memory_mb), 3),
        "words": words,
    }

    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)


def is_missing_char_aligner(error: Exception) -> bool:
    message = str(error).lower()
    if "aligner" in message and ("unexpected" in message or "keyword" in message):
        return True
    if 'aligner must be "new"/"legacy"' in message:
        return True
    if "no module named" in message and "stable_whisper" in message:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper char-level alignment via stable-ts or fallback implementation"
    )
    parser.add_argument("--config", required=True, help="Path to benchmark config.json")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for benchmark output JSON files",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = config.get("audio_files")
    if not isinstance(audio_files, dict):
        raise ValueError("config.json missing audio_files object")

    script_dir = Path(__file__).resolve().parent

    try:
        eprint("[whisper-char-align] loading stable-ts large-v3 on CPU")
        aligner_obj: object = StableTSCharAligner()
    except Exception as init_error:
        eprint(
            "[whisper-char-align] stable-ts char aligner unavailable at init; "
            f"falling back to 30stomercury implementation: {init_error}"
        )
        aligner_obj = WhisperCharAlignmentFallback(script_dir)

    for lang in ("en", "ru"):
        item = audio_files.get(lang)
        if not isinstance(item, dict):
            raise ValueError(f"config.audio_files.{lang} missing")
        audio_path = item.get("path")
        transcript = item.get("reference_transcript")
        language = item.get("language", lang)
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(f"config.audio_files.{lang}.path must be a string")
        if not isinstance(transcript, str):
            raise ValueError(
                f"config.audio_files.{lang}.reference_transcript must be a string"
            )
        if not isinstance(language, str):
            raise ValueError(f"config.audio_files.{lang}.language must be a string")

        output_path = output_dir / f"whisper_char_align_{lang}.json"
        eprint(f"[whisper-char-align] processing {lang}: {audio_path}")
        try:
            run_benchmark(aligner_obj, audio_path, language, transcript, output_path)
        except Exception as run_error:
            if isinstance(aligner_obj, StableTSCharAligner) and is_missing_char_aligner(
                run_error
            ):
                eprint(
                    "[whisper-char-align] stable-ts lacks char aligner at runtime; "
                    f"switching to fallback: {run_error}"
                )
                aligner_obj = WhisperCharAlignmentFallback(script_dir)
                run_benchmark(
                    aligner_obj, audio_path, language, transcript, output_path
                )
            else:
                raise
        eprint(f"[whisper-char-align] wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
