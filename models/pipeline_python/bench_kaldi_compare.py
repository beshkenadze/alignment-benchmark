#!/usr/bin/env python3
"""Pure Kaldi vs kalpy comparison: same segments + text, only aligner differs.

Takes Swift pipeline JSON as source of segments/transcripts,
runs Python kalpy alignment on the same data.

Usage:
    python bench_kaldi_compare.py <swift-result.json> --language en|ru
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

eprint = lambda *a, **k: print(*a, **k, file=sys.stderr)

PRETRAINED_ACOUSTIC = Path.home() / "Documents/MFA/pretrained_models/acoustic"
PRETRAINED_DICT = Path.home() / "Documents/MFA/pretrained_models/dictionary"

LANG_SETTINGS = {
    "ru": {
        "acoustic": str(PRETRAINED_ACOUSTIC / "russian_mfa.zip"),
        "dictionary": str(PRETRAINED_DICT / "russian_mfa.dict"),
    },
    "en": {
        "acoustic": str(PRETRAINED_ACOUSTIC / "english_mfa.zip"),
        "dictionary": str(PRETRAINED_DICT / "english_mfa.dict"),
    },
}


def load_audio(path: str) -> tuple[np.ndarray, int]:
    import subprocess
    import soundfile as sf
    import tempfile

    # ffmpeg to 16kHz mono WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", tmp_path],
        capture_output=True,
    )
    samples, sr = sf.read(tmp_path, dtype="float32")
    Path(tmp_path).unlink(missing_ok=True)
    return samples, sr


def run_kalpy_alignment(
    samples: np.ndarray,
    sr: int,
    segments: list[dict],
    settings: dict[str, str],
) -> tuple[list[dict], float, float]:
    from kalpy.feat.cmvn import CmvnComputer
    from kalpy.fstext.lexicon import LexiconCompiler
    from kalpy.utterance import Segment, Utterance as KalpyUtterance
    from kalpy.aligner import KalpyAligner
    from montreal_forced_aligner.models import AcousticModel
    import tempfile
    import soundfile as sf

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

    cmvn_computer = CmvnComputer()
    all_results = []
    total_words = 0
    fail_count = 0

    t1 = time.perf_counter()
    for idx, seg in enumerate(segments):
        start_sample = int(seg["start_time"] * sr)
        end_sample = min(int(seg["end_time"] * sr), len(samples))
        seg_samples = samples[start_sample:end_sample]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, seg_samples, sr)
            tmp_path = f.name

        try:
            seg_obj = Segment(tmp_path, 0, None, 0)
            utt = KalpyUtterance(seg_obj, seg["transcript"].strip().lower())
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
                    "transcript": seg["transcript"],
                    "swift_words": seg["word_count"],
                    "kalpy_words": len(words),
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
                    "transcript": seg["transcript"],
                    "swift_words": seg["word_count"],
                    "kalpy_words": 0,
                    "words": [],
                    "error": str(e),
                }
            )
            if fail_count <= 5:
                eprint(f"  ⚠ Alignment failed for segment {seg['index']}: {e}")

        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if (idx + 1) % 20 == 0 or idx == len(segments) - 1:
            eprint(
                f"  Align progress: {idx + 1}/{len(segments)} segments, {total_words} words"
            )

    inference_time = time.perf_counter() - t1
    eprint(
        f"✓ Alignment: {total_words} words, {fail_count} failures ({inference_time:.2f}s)"
    )
    return all_results, model_load_time, inference_time


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure Kaldi vs kalpy comparison")
    parser.add_argument("swift_json", help="Path to Swift pipeline result JSON")
    parser.add_argument("--language", default="en", choices=["en", "ru"])
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    with open(args.swift_json) as f:
        swift_data = json.load(f)

    audio_file = swift_data["audioFile"]
    audio_duration = swift_data["audioDuration"]
    swift_segments = swift_data["segments"]

    segments = []
    swift_total_words = 0
    for seg in swift_segments:
        word_count = len(seg.get("words", []))
        swift_total_words += word_count
        segments.append(
            {
                "index": seg["index"],
                "start_time": seg["startTime"],
                "end_time": seg["endTime"],
                "transcript": seg["transcript"],
                "word_count": word_count,
            }
        )

    eprint(f"╔══════════════════════════════════════════════╗")
    eprint(f"║  Kaldi vs kalpy: same segments + text        ║")
    eprint(f"╚══════════════════════════════════════════════╝")
    eprint(f"Source: {args.swift_json}")
    eprint(f"Audio: {audio_file}")
    eprint(f"Language: {args.language}")
    eprint(f"Segments: {len(segments)}")
    eprint(f"Swift words (reference): {swift_total_words}")
    eprint()

    t0 = time.perf_counter()
    samples, sr = load_audio(audio_file)
    audio_load_time = time.perf_counter() - t0
    eprint(f"✓ Audio loaded ({audio_load_time:.2f}s)")

    settings = LANG_SETTINGS[args.language]
    results, model_load_time, inference_time = run_kalpy_alignment(
        samples,
        sr,
        segments,
        settings,
    )

    kalpy_total_words = sum(r["kalpy_words"] for r in results)
    kalpy_failures = sum(1 for r in results if "error" in r)

    # Per-segment comparison
    match_count = 0
    swift_more = 0
    kalpy_more = 0
    for r in results:
        sw = r["swift_words"]
        kw = r["kalpy_words"]
        if sw == kw:
            match_count += 1
        elif sw > kw:
            swift_more += 1
        else:
            kalpy_more += 1

    swift_align_load = swift_data.get("stages", {}).get("4_align_load", "N/A")
    swift_align_inf = swift_data.get("stages", {}).get("4_align_inference", "N/A")

    eprint()
    eprint("=== COMPARISON ===")
    eprint(f"  Segments:           {len(segments)} (identical)")
    eprint(f"  Transcripts:        identical (from Swift ASR)")
    eprint(f"  Swift words:        {swift_total_words}")
    eprint(f"  kalpy words:        {kalpy_total_words}")
    eprint(
        f"  Word diff:          {kalpy_total_words - swift_total_words:+d} ({(kalpy_total_words - swift_total_words) / swift_total_words * 100:+.1f}%)"
    )
    eprint(
        f"  Segments match:     {match_count}/{len(segments)} ({match_count / len(segments) * 100:.1f}%)"
    )
    eprint(f"  Swift has more:     {swift_more}")
    eprint(f"  kalpy has more:     {kalpy_more}")
    eprint(f"  kalpy failures:     {kalpy_failures}")
    eprint()
    eprint("=== TIMING ===")
    eprint(f"  Swift align load:   {swift_align_load}s")
    eprint(f"  kalpy align load:   {model_load_time:.2f}s")
    eprint(f"  Swift align infer:  {swift_align_inf}s")
    eprint(f"  kalpy align infer:  {inference_time:.2f}s")

    output = {
        "source": args.swift_json,
        "audioFile": audio_file,
        "audioDuration": audio_duration,
        "language": args.language,
        "segmentCount": len(segments),
        "swift_words": swift_total_words,
        "kalpy_words": kalpy_total_words,
        "word_diff": kalpy_total_words - swift_total_words,
        "segments_word_match": match_count,
        "swift_has_more_words": swift_more,
        "kalpy_has_more_words": kalpy_more,
        "kalpy_failures": kalpy_failures,
        "swift_align_load": swift_align_load,
        "kalpy_align_load": model_load_time,
        "swift_align_inference": swift_align_inf,
        "kalpy_align_inference": inference_time,
        "segments": results,
    }

    output_path = args.output or str(
        Path(__file__).parent.parent.parent
        / "results"
        / f"kaldi_compare_{args.language}.json"
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    eprint(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
