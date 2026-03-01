#!/usr/bin/env python3
"""
MFA v3 Deep-Dive Profiling Script.

Tests multiple configurations to find optimal performance settings:
1. Baseline (current defaults: blas_num_threads=1, num_jobs=3, use_threading=true)
2. Increased BLAS threads (match CPU cores)
3. Single speaker mode (disables speaker adaptation)
4. Increased num_jobs
5. JSON output format (skip TextGrid overhead)
6. Combined optimizations

Outputs timing breakdown for each stage.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TimingResult:
    config_name: str
    language: str
    audio_prep_seconds: float = 0.0
    alignment_seconds: float = 0.0
    total_seconds: float = 0.0
    word_count: int = 0
    extra: dict = field(default_factory=dict)


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def convert_to_wav16k(source: Path, dest: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-ar", "16000", "-ac", "1", str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )


def count_words_in_textgrid(tg_path: Path) -> int:
    import re

    text = tg_path.read_text(encoding="utf-8", errors="replace")
    in_words = False
    count = 0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("name = "):
            in_words = line == 'name = "words"'
        if in_words and line.startswith("text = "):
            match = re.match(r'text = "(.*)"$', line)
            if match and match.group(1).strip():
                count += 1
    return count


def count_words_in_json(json_path: Path) -> int:
    data = json.loads(json_path.read_text())
    count = 0
    for utt in data.get("tiers", {}).get("words", {}).get("entries", []):
        if isinstance(utt, list) and len(utt) >= 3 and utt[2].strip():
            count += 1
    # Fallback: try different JSON structure
    if count == 0:
        for tier_name, tier_data in data.get("tiers", {}).items():
            if "word" in tier_name.lower():
                entries = tier_data.get("entries", [])
                count = sum(
                    1
                    for e in entries
                    if isinstance(e, list) and len(e) >= 3 and e[2].strip()
                )
                break
    return count


def run_mfa_alignment(
    *,
    language: str,
    audio_path: Path,
    transcript: str,
    acoustic_model: str,
    dictionary: str,
    config_name: str,
    extra_flags: list[str],
    output_format: str = "long_textgrid",
) -> TimingResult:
    result = TimingResult(config_name=config_name, language=language)

    with tempfile.TemporaryDirectory(prefix=f"mfa_profile_{language}_") as tmp:
        tmp_root = Path(tmp)
        corpus_dir = tmp_root / "corpus"
        output_dir = tmp_root / "aligned"
        corpus_dir.mkdir()
        output_dir.mkdir()

        # Audio prep timing
        t0 = time.perf_counter()
        corpus_audio = corpus_dir / f"sample_{language}.wav"
        corpus_txt = corpus_dir / f"sample_{language}.txt"
        convert_to_wav16k(audio_path, corpus_audio)
        corpus_txt.write_text(transcript.strip() + "\n", encoding="utf-8")
        result.audio_prep_seconds = time.perf_counter() - t0

        # Alignment timing
        cmd = [
            "conda",
            "run",
            "-n",
            "mfa_bench",
            "mfa",
            "align",
            str(corpus_dir),
            dictionary,
            acoustic_model,
            str(output_dir),
            "--clean",
            "--output_format",
            output_format,
            "--quiet",
        ] + extra_flags

        eprint(f"\n{'=' * 60}")
        eprint(f"[profile] Config: {config_name} | Language: {language}")
        eprint(
            f"[profile] Flags: {' '.join(extra_flags) if extra_flags else '(defaults)'}"
        )
        eprint(f"[profile] Running: {' '.join(cmd)}")

        t1 = time.perf_counter()
        completed = subprocess.run(cmd, capture_output=True, text=True)
        result.alignment_seconds = time.perf_counter() - t1

        if completed.returncode != 0:
            eprint(f"[profile] FAILED! Return code: {completed.returncode}")
            eprint(f"[profile] stderr: {completed.stderr[-500:]}")
            result.extra["error"] = completed.stderr[-500:]
            return result

        result.total_seconds = result.audio_prep_seconds + result.alignment_seconds

        # Count output words
        if output_format == "json":
            jsons = sorted(output_dir.glob("*.json"))
            if jsons:
                result.word_count = count_words_in_json(jsons[0])
        else:
            tgs = sorted(output_dir.glob("*.TextGrid"))
            if tgs:
                result.word_count = count_words_in_textgrid(tgs[0])

        eprint(
            f"[profile] Done: {result.alignment_seconds:.2f}s alignment, {result.word_count} words"
        )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MFA v3 profiling across configurations"
    )
    parser.add_argument("--config", required=True, help="Path to benchmark config.json")
    parser.add_argument("--language", default="ru", choices=["en", "ru", "both"])
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())

    LANG_SETTINGS = {
        "en": {"acoustic": "english_mfa", "dict": "english_mfa"},
        "ru": {"acoustic": "russian_mfa", "dict": "russian_mfa"},
    }

    languages = ["en", "ru"] if args.language == "both" else [args.language]

    # Define test configurations
    CONFIGS: list[tuple[str, list[str]]] = [
        ("baseline", []),
        ("single_speaker", ["--single_speaker"]),
        ("jobs_1", ["-j", "1"]),
        ("jobs_8", ["-j", "8"]),
        ("no_mp", ["--no_use_mp"]),
        ("single_speaker_j1", ["--single_speaker", "-j", "1"]),
        ("single_speaker_j8", ["--single_speaker", "-j", "8"]),
    ]

    all_results: list[dict] = []

    for lang in languages:
        settings = LANG_SETTINGS[lang]
        audio_entry = config["audio_files"][lang]
        audio_path = Path(audio_entry["path"]).expanduser().resolve()
        transcript = audio_entry["reference_transcript"]

        if not audio_path.exists():
            eprint(f"Audio file not found: {audio_path}")
            continue

        for config_name, flags in CONFIGS:
            try:
                result = run_mfa_alignment(
                    language=lang,
                    audio_path=audio_path,
                    transcript=transcript,
                    acoustic_model=settings["acoustic"],
                    dictionary=settings["dict"],
                    config_name=config_name,
                    extra_flags=flags,
                )
                all_results.append(
                    {
                        "config": result.config_name,
                        "language": result.language,
                        "audio_prep_s": round(result.audio_prep_seconds, 3),
                        "alignment_s": round(result.alignment_seconds, 3),
                        "total_s": round(result.total_seconds, 3),
                        "word_count": result.word_count,
                        "error": result.extra.get("error"),
                    }
                )
            except Exception as e:
                eprint(f"[profile] Exception for {config_name}/{lang}: {e}")
                all_results.append(
                    {
                        "config": config_name,
                        "language": lang,
                        "error": str(e),
                    }
                )

    # Also test with different blas_num_threads via env var
    for blas_threads in [2, 4, 8]:
        for lang in languages:
            settings = LANG_SETTINGS[lang]
            audio_entry = config["audio_files"][lang]
            audio_path = Path(audio_entry["path"]).expanduser().resolve()
            transcript = audio_entry["reference_transcript"]

            config_name = f"blas_{blas_threads}t_single"

            with tempfile.TemporaryDirectory(prefix=f"mfa_blas_{lang}_") as tmp:
                tmp_root = Path(tmp)
                corpus_dir = tmp_root / "corpus"
                output_dir = tmp_root / "aligned"
                corpus_dir.mkdir()
                output_dir.mkdir()

                corpus_audio = corpus_dir / f"sample_{lang}.wav"
                corpus_txt = corpus_dir / f"sample_{lang}.txt"
                convert_to_wav16k(audio_path, corpus_audio)
                corpus_txt.write_text(transcript.strip() + "\n", encoding="utf-8")

                env = os.environ.copy()
                env["OPENBLAS_NUM_THREADS"] = str(blas_threads)
                env["MKL_NUM_THREADS"] = str(blas_threads)
                env["OMP_NUM_THREADS"] = str(blas_threads)

                cmd = [
                    "conda",
                    "run",
                    "-n",
                    "mfa_bench",
                    "mfa",
                    "align",
                    str(corpus_dir),
                    settings["dict"],
                    settings["acoustic"],
                    str(output_dir),
                    "--clean",
                    "--quiet",
                    "--single_speaker",
                ]

                eprint(f"\n{'=' * 60}")
                eprint(f"[profile] Config: {config_name} | Language: {lang}")
                eprint(f"[profile] BLAS threads: {blas_threads}")

                t0 = time.perf_counter()
                completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
                elapsed = time.perf_counter() - t0

                word_count = 0
                error = None
                if completed.returncode != 0:
                    error = completed.stderr[-500:]
                else:
                    tgs = sorted(output_dir.glob("*.TextGrid"))
                    if tgs:
                        word_count = count_words_in_textgrid(tgs[0])

                eprint(f"[profile] Done: {elapsed:.2f}s, {word_count} words")
                all_results.append(
                    {
                        "config": config_name,
                        "language": lang,
                        "alignment_s": round(elapsed, 3),
                        "word_count": word_count,
                        "error": error,
                    }
                )

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Config':<28} {'Lang':>4} {'Align(s)':>10} {'Words':>6} {'Status':>8}")
    print("-" * 80)
    for r in all_results:
        status = "OK" if not r.get("error") else "FAIL"
        align = r.get("alignment_s", r.get("total_s", 0))
        print(
            f"{r['config']:<28} {r['language']:>4} {align:>10.2f} {r.get('word_count', 0):>6} {status:>8}"
        )
    print("=" * 80)

    # Save results
    output_path = (
        Path(args.output)
        if args.output
        else Path("/Volumes/DATA/alignment-benchmark/results/mfa_profile.json")
    )
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n")
    eprint(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
