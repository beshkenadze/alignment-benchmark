#!/usr/bin/env python3
"""
MFA v3 Direct Python API Benchmark (Pattern B — KalpyAligner).

Bypasses all CLI/subprocess/DB/TextGrid overhead by using KalpyAligner directly.
This measures the true alignment speed without framework overhead.

Timing breakdown:
1. Model load (one-time)
2. Lexicon FST compilation (one-time, cacheable)
3. MFCC feature extraction (per-utterance)
4. Alignment (per-utterance)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

MFA_DIR = Path.home() / "Documents" / "MFA"
PRETRAINED_ACOUSTIC = MFA_DIR / "pretrained_models" / "acoustic"
PRETRAINED_DICT = MFA_DIR / "pretrained_models" / "dictionary"


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


LANG_SETTINGS: dict[str, dict[str, str]] = {
    "en": {
        "acoustic": str(PRETRAINED_ACOUSTIC / "english_mfa.zip"),
        "dictionary": str(PRETRAINED_DICT / "english_mfa.dict"),
        "json_name": "mfa_api_en.json",
        "model_name": "mfa v3 api (english_mfa)",
    },
    "ru": {
        "acoustic": str(PRETRAINED_ACOUSTIC / "russian_mfa.zip"),
        "dictionary": str(PRETRAINED_DICT / "russian_mfa.dict"),
        "json_name": "mfa_api_ru.json",
        "model_name": "mfa v3 api (russian_mfa)",
    },
}


def align_with_api(
    language: str,
    audio_path: str,
    transcript: str,
    settings: dict[str, str],
) -> dict[str, Any]:
    # Lazy imports to measure import time
    t_import_start = time.perf_counter()

    from kalpy.feat.cmvn import CmvnComputer
    from kalpy.fstext.lexicon import LexiconCompiler
    from kalpy.utterance import Segment, Utterance as KalpyUtterance
    from montreal_forced_aligner.models import AcousticModel

    t_import = time.perf_counter() - t_import_start
    eprint(f"[api-bench] Import time: {t_import:.3f}s")

    # Phase 1: Load acoustic model
    t_model_start = time.perf_counter()
    acoustic_model = AcousticModel(settings["acoustic"])
    t_model = time.perf_counter() - t_model_start
    eprint(f"[api-bench] Acoustic model load: {t_model:.3f}s")

    # Phase 2: Build lexicon FST
    t_lexicon_start = time.perf_counter()
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
    t_lexicon = time.perf_counter() - t_lexicon_start
    eprint(f"[api-bench] Lexicon FST compilation: {t_lexicon:.3f}s")

    # Phase 3: Create KalpyAligner
    from kalpy.aligner import KalpyAligner

    t_aligner_start = time.perf_counter()
    kalpy_aligner = KalpyAligner(
        acoustic_model,
        lexicon_compiler,
        beam=10,
        retry_beam=40,
        acoustic_scale=0.1,
        transition_scale=1.0,
        self_loop_scale=0.1,
    )
    t_aligner = time.perf_counter() - t_aligner_start
    eprint(f"[api-bench] KalpyAligner creation: {t_aligner:.3f}s")

    # Phase 4: Prepare utterance (MFCC extraction)
    t_feat_start = time.perf_counter()
    seg = Segment(audio_path, 0, None, 0)
    utt = KalpyUtterance(seg, transcript.strip().lower())
    utt.generate_mfccs(acoustic_model.mfcc_computer)

    cmvn_computer = CmvnComputer()
    cmvn = cmvn_computer.compute_cmvn_from_features([utt.mfccs])
    utt.apply_cmvn(cmvn)
    t_feat = time.perf_counter() - t_feat_start
    eprint(f"[api-bench] Feature extraction (MFCC+CMVN): {t_feat:.3f}s")

    # Phase 5: Align
    t_align_start = time.perf_counter()
    ctm = kalpy_aligner.align_utterance(utt)
    t_align = time.perf_counter() - t_align_start
    eprint(f"[api-bench] Alignment: {t_align:.3f}s")

    # Extract word intervals
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
                        "start": float(begin),
                        "end": float(end),
                        "confidence": None,
                    }
                )
    eprint(f"[api-bench] Extracted {len(words)} words")

    total_model_load = t_import + t_model + t_lexicon + t_aligner
    total_inference = t_feat + t_align

    return {
        "model": settings["model_name"],
        "audio_file": audio_path,
        "language": language,
        "inference_time_seconds": round(total_inference, 6),
        "model_load_time_seconds": round(total_model_load, 6),
        "peak_memory_mb": 0.0,
        "words": words,
        "timing_breakdown": {
            "import_s": round(t_import, 4),
            "acoustic_model_load_s": round(t_model, 4),
            "lexicon_fst_compile_s": round(t_lexicon, 4),
            "aligner_creation_s": round(t_aligner, 4),
            "feature_extraction_s": round(t_feat, 4),
            "alignment_s": round(t_align, 4),
            "total_one_time_s": round(total_model_load, 4),
            "total_per_utterance_s": round(total_inference, 4),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MFA v3 Python API benchmark")
    parser.add_argument("--config", required=True, help="Path to benchmark config.json")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )
    parser.add_argument("--language", default="both", choices=["en", "ru", "both"])
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = ["en", "ru"] if args.language == "both" else [args.language]

    for lang in languages:
        settings = LANG_SETTINGS[lang]
        audio_entry = config["audio_files"][lang]
        audio_path = Path(audio_entry["path"]).expanduser().resolve()

        if not audio_path.exists():
            eprint(f"Audio not found: {audio_path}")
            continue

        eprint(f"\n{'=' * 60}")
        eprint(f"[api-bench] Language: {lang} | Audio: {audio_path.name}")
        eprint(f"{'=' * 60}")

        try:
            result = align_with_api(
                language=lang,
                audio_path=str(audio_path),
                transcript=audio_entry["reference_transcript"],
                settings=settings,
            )

            out_path = output_dir / settings["json_name"]
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            eprint(f"[api-bench] Saved to {out_path}")

            # Print timing summary
            tb = result["timing_breakdown"]
            eprint(f"\n--- Timing Summary ({lang}) ---")
            eprint(f"  One-time setup:    {tb['total_one_time_s']:.3f}s")
            eprint(f"    Import:          {tb['import_s']:.3f}s")
            eprint(f"    Model load:      {tb['acoustic_model_load_s']:.3f}s")
            eprint(f"    Lexicon FST:     {tb['lexicon_fst_compile_s']:.3f}s")
            eprint(f"    Aligner create:  {tb['aligner_creation_s']:.3f}s")
            eprint(f"  Per-utterance:     {tb['total_per_utterance_s']:.3f}s")
            eprint(f"    Features (MFCC): {tb['feature_extraction_s']:.3f}s")
            eprint(f"    Alignment:       {tb['alignment_s']:.3f}s")
            eprint(f"  Words found:       {len(result['words'])}")

        except Exception as e:
            eprint(f"[api-bench] FAILED for {lang}: {e}")
            import traceback

            traceback.print_exc(file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
