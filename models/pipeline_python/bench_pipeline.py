#!/usr/bin/env python3
"""
Python Pipeline Benchmark: VAD → ASR → Forced Alignment.

Mirror of Swift IntegrationBench pipeline for comparison.
- VAD: Silero VAD (PyTorch CPU)
- ASR: Whisper large-v3-turbo (mlx-audio, MLX GPU)
- Alignment: Kaldi via kalpy (CPU)

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE pixi run python bench_pipeline.py \
        "/Users/akira/Downloads/Spoon Episodex/Episode 171/thereisnospoon-171.mp3" \
        --language ru
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# --- Config ---

MFA_DIR = Path.home() / "Documents" / "MFA"
PRETRAINED_ACOUSTIC = MFA_DIR / "pretrained_models" / "acoustic"
PRETRAINED_DICT = MFA_DIR / "pretrained_models" / "dictionary"

LANG_SETTINGS = {
    "ru": {
        "acoustic": str(PRETRAINED_ACOUSTIC / "russian_mfa.zip"),
        "dictionary": str(PRETRAINED_DICT / "russian_mfa.dict"),
        "language": "Russian",
        "whisper_lang": "ru",
    },
    "en": {
        "acoustic": str(PRETRAINED_ACOUSTIC / "english_mfa.zip"),
        "dictionary": str(PRETRAINED_DICT / "english_mfa.dict"),
        "language": "English",
        "whisper_lang": "en",
    },
}

OUTPUT_DIR = Path("/Volumes/DATA/alignment-benchmark/results")


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_audio_16k(audio_path: str) -> tuple[np.ndarray, int]:
    """Load audio file and resample to 16kHz mono float32."""
    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32")
    # Mix to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != 16000:
        import torchaudio

        import torch

        tensor = torch.from_numpy(data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        tensor = resampler(tensor)
        data = tensor.squeeze(0).numpy()
        sr = 16000

    return data, sr


def run_vad(samples: np.ndarray, sr: int) -> list[dict[str, Any]]:
    """Run Silero VAD and return speech segments."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    wav = torch.from_numpy(samples)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        max_speech_duration_s=30.0,
        speech_pad_ms=150,
    )

    segments = []
    for ts in speech_timestamps:
        start_sample = ts["start"]
        end_sample = ts["end"]
        segments.append(
            {
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_time": start_sample / sr,
                "end_time": end_sample / sr,
                "duration": (end_sample - start_sample) / sr,
            }
        )

    return segments


def run_asr(
    samples: np.ndarray,
    sr: int,
    vad_segments: list[dict],
    language: str,
) -> list[dict[str, Any]]:
    """Run ASR on each VAD segment using mlx-audio Whisper."""
    from mlx_audio.stt.generate import generate_transcription
    from mlx_audio.stt.utils import load_model
    import tempfile
    import soundfile as sf

    eprint("Loading Whisper large-v3-turbo model...")
    t0 = time.perf_counter()
    model = load_model("mlx-community/whisper-large-v3-turbo")
    model_load_time = time.perf_counter() - t0
    eprint(f"  Model loaded in {model_load_time:.2f}s")

    transcripts = []
    for idx, seg in enumerate(vad_segments):
        seg_samples = samples[seg["start_sample"] : seg["end_sample"]]

        # Write segment to temp file (mlx-audio needs file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, seg_samples, sr)
            tmp_path = f.name

        try:
            # Use temp dir for output (generate_transcription appends .txt)
            with tempfile.TemporaryDirectory() as tmp_dir:
                out_path = str(Path(tmp_dir) / "output")
                result = generate_transcription(
                    model=model,
                    audio_path=tmp_path,
                    output_path=out_path,
                    format="txt",
                    verbose=False,
                    language=language,
                )
                text = (
                    result.text.strip() if hasattr(result, "text") else str(result).strip()
                )
        except Exception as e:
            eprint(f"  ⚠ ASR failed for segment {idx}: {e}")
            text = ""
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if text:
            transcripts.append(
                {
                    "index": idx,
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "text": text,
                }
            )

        if (idx + 1) % 10 == 0 or idx == len(vad_segments) - 1:
            eprint(f"  ASR progress: {idx + 1}/{len(vad_segments)} segments")

    return transcripts, model_load_time


def run_alignment(
    samples: np.ndarray,
    sr: int,
    transcripts: list[dict],
    settings: dict[str, str],
) -> tuple[list[dict], float]:
    """Run Kaldi forced alignment on each transcribed segment."""
    from kalpy.feat.cmvn import CmvnComputer
    from kalpy.fstext.lexicon import LexiconCompiler
    from kalpy.utterance import Segment, Utterance as KalpyUtterance
    from kalpy.aligner import KalpyAligner
    from montreal_forced_aligner.models import AcousticModel
    import tempfile
    import soundfile as sf

    # Load model
    t0 = time.perf_counter()
    acoustic_model = AcousticModel(settings["acoustic"])
    params = acoustic_model.parameters

    lexicon_compiler = LexiconCompiler(
        disambiguation=False,
        silence_probability=params.get("silence_probability", 0.5),
        initial_silence_probability=params.get("initial_silence_probability", 0.5),
        final_silence_correction=params.get("final_silence_correction", None),
        final_non_silence_correction=params.get("final_non_silence_correction", None),
        silence_phone=params.get("optional_silence_phone", "sil"),
        oov_phone=params.get("oov_phone", "spn"),
        position_dependent_phones=params.get("position_dependent_phones", True),
        phones=params.get("non_silence_phones", []),
    )
    lexicon_compiler.load_pronunciations(settings["dictionary"])
    lexicon_compiler.create_fsts()
    lexicon_compiler.clear()

    kalpy_aligner = KalpyAligner(
        acoustic_model,
        lexicon_compiler,
        beam=10,
        retry_beam=40,
        acoustic_scale=0.1,
        transition_scale=1.0,
        self_loop_scale=0.1,
    )
    model_load_time = time.perf_counter() - t0
    eprint(f"  Alignment model loaded in {model_load_time:.2f}s")

    # Align each segment
    cmvn_computer = CmvnComputer()
    all_results = []
    total_words = 0
    fail_count = 0

    for idx, seg in enumerate(transcripts):
        start_sample = int(seg["start_time"] * sr)
        end_sample = min(int(seg["end_time"] * sr), len(samples))
        seg_samples = samples[start_sample:end_sample]

        # Write segment to temp wav for kalpy
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, seg_samples, sr)
            tmp_path = f.name

        try:
            seg_obj = Segment(tmp_path, 0, None, 0)
            utt = KalpyUtterance(seg_obj, seg["text"].strip().lower())
            utt.generate_mfccs(acoustic_model.mfcc_computer)
            cmvn = cmvn_computer.compute_cmvn_from_features([utt.mfccs])
            utt.apply_cmvn(cmvn)

            ctm = kalpy_aligner.align_utterance(utt)

            words = []
            if ctm is not None and hasattr(ctm, "word_intervals"):
                for wi in ctm.word_intervals:
                    label = wi.label if hasattr(wi, "label") else str(wi)
                    begin = wi.begin if hasattr(wi, "begin") else 0.0
                    end = wi.end if hasattr(wi, "end") else 0.0
                    if label.strip() and label not in ("<eps>", "sil", "sp", "spn"):
                        words.append(
                            {
                                "word": label,
                                "start": float(begin) + seg["start_time"],
                                "end": float(end) + seg["start_time"],
                            }
                        )

            all_results.append(
                {
                    "index": seg["index"],
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "transcript": seg["text"],
                    "words": words,
                }
            )
            total_words += len(words)

        except Exception as e:
            fail_count += 1
            all_results.append(
                {
                    "index": seg["index"],
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "transcript": seg["text"],
                    "words": [],
                }
            )
            if fail_count <= 5:
                eprint(f"  ⚠ Alignment failed for segment {seg['index']}: {e}")

        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if (idx + 1) % 20 == 0 or idx == len(transcripts) - 1:
            eprint(
                f"  Align progress: {idx + 1}/{len(transcripts)} segments, {total_words} words"
            )

    eprint(f"✓ Alignment: {total_words} words, {fail_count} failures")
    return all_results, model_load_time


def main() -> int:
    parser = argparse.ArgumentParser(description="Python Pipeline Benchmark")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--language", default="ru", choices=["en", "ru"])
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    settings = LANG_SETTINGS[args.language]
    audio_path = args.audio

    eprint("╔══════════════════════════════════════════════╗")
    eprint("║  Python Pipeline: VAD → ASR → Align         ║")
    eprint("╚══════════════════════════════════════════════╝")
    eprint(f"Audio: {audio_path}")
    eprint(f"Language: {settings['language']}")
    eprint("")

    timings = {}

    # Stage 1: Load Audio
    t0 = time.perf_counter()
    samples, sr = load_audio_16k(audio_path)
    audio_duration = len(samples) / sr
    timings["1_audio_load"] = time.perf_counter() - t0
    eprint(
        f"✓ Audio loaded: {audio_duration:.1f}s, {sr}Hz, {len(samples)} samples ({timings['1_audio_load']:.2f}s)"
    )

    # Stage 2: VAD
    t0 = time.perf_counter()
    vad_segments = run_vad(samples, sr)
    timings["2_vad"] = time.perf_counter() - t0
    eprint(f"✓ VAD: {len(vad_segments)} speech segments ({timings['2_vad']:.2f}s)")
    for seg in vad_segments[:5]:
        eprint(
            f"  [{vad_segments.index(seg)}] {seg['start_time']:.2f}s - {seg['end_time']:.2f}s ({seg['duration']:.1f}s)"
        )
    if len(vad_segments) > 5:
        eprint(f"  ... and {len(vad_segments) - 5} more")

    # Stage 3: ASR
    t0 = time.perf_counter()
    transcripts, asr_model_load = run_asr(
        samples, sr, vad_segments, settings["whisper_lang"]
    )
    total_asr_time = time.perf_counter() - t0
    timings["3_asr_load"] = asr_model_load
    timings["3_asr_inference"] = total_asr_time - asr_model_load
    eprint(f"✓ ASR: {len(transcripts)} segments transcribed ({total_asr_time:.2f}s)")

    # Stage 4: Alignment
    t0 = time.perf_counter()
    align_results, align_model_load = run_alignment(samples, sr, transcripts, settings)
    total_align_time = time.perf_counter() - t0
    timings["4_align_load"] = align_model_load
    timings["4_align_inference"] = total_align_time - align_model_load

    total_words = sum(len(r["words"]) for r in align_results)

    # Report
    total_time = sum(timings.values())
    rtfx = audio_duration / total_time if total_time > 0 else 0

    eprint("\n=== TIMING REPORT ===")
    for name, duration in timings.items():
        pct = (duration / total_time) * 100 if total_time > 0 else 0
        eprint(f"  {name}: {duration:.2f}s ({pct:.1f}%)")
    eprint("  ─────────────────────")
    eprint(f"  TOTAL: {total_time:.2f}s")
    eprint(f"  Audio duration: {audio_duration:.1f}s")
    eprint(f"  RTFx: {rtfx:.1f}x realtime")

    result = {
        "pipeline": "python",
        "asr_model": "whisper-large-v3-turbo (mlx-audio)",
        "vad_model": "silero-vad (PyTorch CPU)",
        "align_model": "kaldi (kalpy/MFA)",
        "audioFile": audio_path,
        "audioDuration": round(audio_duration, 3),
        "totalTime": round(total_time, 3),
        "rtfx": round(rtfx, 3),
        "stages": {k: round(v, 3) for k, v in timings.items()},
        "segmentCount": len(align_results),
        "wordCount": total_words,
        "segments": align_results,
    }

    output_path = args.output or str(
        OUTPUT_DIR / f"pipeline_python_{args.language}.json"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    eprint(f"\nResults saved to: {output_path}")

    # Sample output
    eprint("\n=== SAMPLE OUTPUT (first 3 segments) ===")
    for seg in align_results[:3]:
        eprint(f"\n[{seg['index']}] {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")
        eprint(f"  Text: {seg['transcript'][:80]}...")
        for w in seg["words"][:5]:
            eprint(f"  {w['start']:7.3f} - {w['end']:7.3f}  {w['word']}")
        if len(seg["words"]) > 5:
            eprint(f"  ... +{len(seg['words']) - 5} more words")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
