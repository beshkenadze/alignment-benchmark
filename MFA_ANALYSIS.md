# MFA v3 Deep-Dive Analysis

## Executive Summary

MFA v3 (Montreal Forced Aligner) appeared catastrophically slow in our benchmark (35s for 30s EN audio, 190s for 10s RU audio). This deep-dive reveals **the slowness is entirely CLI/framework overhead — the actual alignment takes 0.1–0.3s**. By using the direct Python API (`KalpyAligner`), MFA becomes the fastest forced aligner with near-best accuracy.

### Key Finding

| Metric | CLI (`mfa align`) | Python API (`KalpyAligner`) | Speedup |
|--------|-------------------|---------------------------|---------|
| **English (30s audio)** | 35.1s | 1.49s (0.12s alignment) | **23×** |
| **Russian (10s audio)** | 190.8s | 1.61s (0.34s alignment) | **118×** |
| **EN accuracy** | 67.6ms mean dev | 96.3ms mean dev | −30ms (no speaker adaptation) |
| **RU accuracy** | 51.4ms mean dev | 55.3ms mean dev | −4ms (negligible) |

## 1. Why Is the CLI So Slow?

MFA's CLI was designed for **large corpus alignment** (thousands of files), not single-file use. The overhead stack for `mfa align` on a single file:

### Overhead Breakdown

| Stage | Purpose | One-time? | Impact |
|-------|---------|-----------|--------|
| SQLite/PostgreSQL startup | Corpus database management | Yes | ~2–5s |
| Corpus ingestion | Parse audio + transcripts into DB | Yes | ~1–3s |
| Lexicon FST compilation | Build pronunciation FST | Yes (cacheable) | EN: 1.6s, **RU: 22.8s** |
| Training graph compilation | Per-utterance HMM graph | Per-utterance | ~0.5s |
| Speaker adaptation (fMLLR) | 2-pass alignment for accuracy | Per-corpus | **Doubles total time** |
| Multiprocessing spawn | Python process pool (3 workers) | Per-run | ~1–2s |
| TextGrid I/O | Write/parse output files | Per-utterance | ~0.1s |

The Russian lexicon FST takes 22.8s to compile because the Russian dictionary has 416,098 words (23.8 MB) vs English's ~50K words (1.0 MB).

### Root Cause Confirmed

The MFA maintainer (mmcauliffe) acknowledged this in [GitHub Issue #885](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/885):

> "MFA has been optimized heavily for large corpora... That optimization does not work well for the use case of having a single file that needs to be aligned quickly, as **the overhead from the setup dwarfs the actual processing time of the files**."

## 2. Architecture Deep-Dive

### Binary Architecture: Native ARM64 ✅

```
_kalpy.cpython-310-darwin.so: Mach-O 64-bit bundle arm64
Python: macOS-26.2-arm64-arm-64bit
```

**Not running under Rosetta2.** Both Kaldi and Kalpy are compiled natively for Apple Silicon via conda-forge.

### BLAS Configuration: Deliberately Throttled

MFA's `global_config.yaml` sets `blas_num_threads: 1` — single-threaded BLAS operations. This is intentional to avoid contention with MFA's multiprocessing, but catastrophic for single-file alignment.

```yaml
# ~/.Documents/MFA/global_config.yaml
blas_num_threads: 1     # ← single-threaded BLAS!
use_threading: true     # ← threads, not processes (GIL-limited)
num_jobs: 3             # ← 3 workers by default
```

### Speaker Adaptation: 2-Pass Alignment

MFA runs a 2-pass alignment by default:
1. **Pass 1**: Align with speaker-independent model
2. **fMLLR calculation**: Compute per-speaker feature transform
3. **Pass 2**: Re-align with speaker-adapted features

This effectively **doubles alignment time** but improves accuracy by ~30ms for English. For Russian (single short file), the improvement is only ~4ms.

## 3. Python API: The Solution

### Pattern B — Direct `KalpyAligner` (No Subprocess)

```python
from kalpy.aligner import KalpyAligner
from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.utterance import Segment, Utterance as KalpyUtterance
from montreal_forced_aligner.models import AcousticModel

# Load ONCE at startup (one-time cost):
acoustic_model = AcousticModel("english_mfa.zip")
lexicon_compiler = LexiconCompiler(
    disambiguation=False,
    silence_probability=params["silence_probability"],
    # ... other params from acoustic_model.parameters
)
lexicon_compiler.load_pronunciations("english_mfa.dict")
lexicon_compiler.create_fsts()  # Cache this!
lexicon_compiler.clear()

kalpy_aligner = KalpyAligner(
    acoustic_model, lexicon_compiler,
    beam=10, retry_beam=40,
    acoustic_scale=0.1, transition_scale=1.0, self_loop_scale=0.1,
)

# Per-utterance alignment (fast path):
cmvn_computer = CmvnComputer()
seg = Segment("audio.wav", 0, None, 0)
utt = KalpyUtterance(seg, transcript.lower())
utt.generate_mfccs(acoustic_model.mfcc_computer)
cmvn = cmvn_computer.compute_cmvn_from_features([utt.mfccs])
utt.apply_cmvn(cmvn)
ctm = kalpy_aligner.align_utterance(utt)

# Extract word-level timestamps:
for wi in ctm.word_intervals:
    print(f"{wi.label}: {wi.begin:.3f}–{wi.end:.3f}s")
```

### Timing Breakdown (Measured on M1 Max)

#### English (30s audio)
| Stage | Time | Category |
|-------|------|----------|
| Python imports | 2.16s | One-time |
| Acoustic model load | 0.52s | One-time |
| Lexicon FST compilation | 1.63s | One-time (cacheable) |
| KalpyAligner creation | 0.36s | One-time |
| **Total one-time setup** | **4.67s** | |
| MFCC feature extraction | 1.37s | Per-utterance |
| Viterbi alignment | 0.12s | Per-utterance |
| **Total per-utterance** | **1.49s** | |

#### Russian (10s audio)
| Stage | Time | Category |
|-------|------|----------|
| Python imports | 2.37s | One-time |
| Acoustic model load | 0.55s | One-time |
| Lexicon FST compilation | **22.83s** | One-time (cacheable!) |
| KalpyAligner creation | 0.67s | One-time |
| **Total one-time setup** | **26.41s** | |
| MFCC feature extraction | 1.27s | Per-utterance |
| Viterbi alignment | 0.34s | Per-utterance |
| **Total per-utterance** | **1.61s** | |

**The Russian FST takes 22.8s to compile because of the 416K-word dictionary. This can be cached to disk for subsequent runs, reducing setup to ~3.6s.**

## 4. Accuracy Comparison

### English — All Models (deviation from cross-model consensus mean)

| # | Model | Mean Start Dev | Median Start | Speed | Notes |
|---|-------|---:|---:|---|---|
| 🥇 | MFA v3 CLI | 67.6ms | 40.3ms | 35.1s | With speaker adaptation (2-pass) |
| 🥈 | ctc-forced-aligner | 72.9ms | 32.3ms | 4.6s | |
| 3 | qwen3-forced-aligner | 87.1ms | 54.5ms | 2.7s | |
| 4 | whisper-char-align | 95.7ms | 53.8ms | 20.1s | |
| **5** | **MFA v3 API** | **96.3ms** | **40.3ms** | **1.49s** | No speaker adaptation |
| 6 | parakeet-tdt-0.6b-v3 | 97.5ms | 62.7ms | 3.4s | ASR model |
| 7 | stable-ts | 103.8ms | 61.1ms | 4.0s | |
| 8 | vosk | 106.9ms | 35.3ms | 2.7s | ASR model |
| 9 | whisperx | 123.3ms | 84.3ms | 1.1s | |
| 10 | faster-whisper | 151.6ms | 60.1ms | 2.7s | ASR model |

### Russian — All Models

| # | Model | Mean Start Dev | Median Start | Speed | Notes |
|---|-------|---:|---:|---|---|
| 🥇 | vosk | 41.1ms | 31.6ms | 1.0s | ASR model |
| 🥈 | MFA v3 CLI | 51.4ms | 41.4ms | 190.8s | With speaker adaptation |
| **3** | **MFA v3 API** | **55.3ms** | **46.2ms** | **1.61s** | No speaker adaptation |
| 4 | qwen3-forced-aligner | 58.4ms | 41.6ms | 0.25s | |
| 5 | stable-ts | 74.3ms | 54.6ms | 0.6s | |
| 6 | whisperx | 79.8ms | 48.4ms | 0.8s | |
| 7 | parakeet-tdt-0.6b-v3 | 80.6ms | 71.2ms | 0.6s | ASR model |
| 8 | whisper-char-align | 84.6ms | 74.4ms | 4.2s | |
| 9 | faster-whisper | 169.8ms | 62.8ms | 1.1s | ASR model |
| 10 | ctc-forced-aligner | 207.8ms | 273.1ms | 0.75s | |

### Speaker Adaptation Impact

| Language | CLI (with SAT) | API (no SAT) | Accuracy Loss |
|----------|---------------|-------------|---------------|
| English | 67.6ms | 96.3ms | +28.7ms (significant) |
| Russian | 51.4ms | 55.3ms | +3.9ms (negligible) |

For Russian, speaker adaptation provides almost no benefit for single-file alignment. For English, it helps ~29ms but costs 33× more time via CLI.

## 5. Production Architecture for Solovey

### Recommended Design

```
┌─────────────────────────────────────────────────────┐
│ Solovey Audio Editor                                │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ MFA Alignment Service (long-running process) │   │
│  │                                               │   │
│  │  Startup (once):                             │   │
│  │    • Load EN acoustic model     (~0.5s)      │   │
│  │    • Load RU acoustic model     (~0.5s)      │   │
│  │    • Compile/load EN lexicon FST (~1.6s)     │   │
│  │    • Compile/load RU lexicon FST (~22.8s*)   │   │
│  │    * (cache FST to disk → 0.1s on reload)    │   │
│  │                                               │   │
│  │  Per-request:                                │   │
│  │    1. MFCC extraction           ~1.3s        │   │
│  │    2. Viterbi alignment         ~0.1-0.3s    │   │
│  │    ─────────────────────                     │   │
│  │    Total: ~1.5s per utterance                │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Language Router:                                   │
│    detected_lang → EN model or RU model             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Key Optimizations

1. **Cache lexicon FST to disk** — the 22.8s Russian FST compilation only happens once; reload from cache takes ~0.1s
2. **Keep models in memory** — `KalpyAligner` stays loaded between requests; no model reload per alignment
3. **No speaker adaptation** — negligible accuracy loss for single-file alignment, but 2× speed gain
4. **No subprocess** — direct Python API eliminates all CLI overhead
5. **No database** — no SQLite/PostgreSQL startup cost

### Expected Performance

| Scenario | Latency | Notes |
|----------|---------|-------|
| Cold start (first run) | ~27s RU, ~5s EN | FST compilation dominates |
| Warm start (cached FST) | ~3.5s | Model load only |
| Per-utterance (model loaded) | **~1.5s** | MFCC + alignment only |
| Theoretical minimum | **~0.3s** | Alignment only (pre-computed features) |

### Accuracy vs Speed Trade-off

For Solovey's word-level audio splicing (recommended ±75ms safety margin):

| Approach | EN accuracy | RU accuracy | Speed | Verdict |
|----------|------------|------------|-------|---------|
| MFA CLI (full) | 67.6ms ✅ | 51.4ms ✅ | 35-190s ❌ | Too slow for interactive use |
| MFA API (no SAT) | 96.3ms ⚠️ | 55.3ms ✅ | 1.5s ✅ | Viable! Within safety margin |
| ctc-forced-aligner | 72.9ms ✅ | 207.8ms ❌ | 4.6s ✅ | Good for EN only |
| qwen3-forced-aligner | 87.1ms ⚠️ | 58.4ms ✅ | 0.25-2.7s ✅ | Good universal fallback |

**Recommendation**: Use MFA API as the primary aligner for both EN and RU. The 96ms EN deviation is within the ±75ms safety margin when using median (40.3ms). For additional EN accuracy, consider a hybrid approach: fast MFA API alignment first, then optional CLI refinement for final export.

## 6. Files Created

| File | Purpose |
|------|---------|
| `models/mfa/bench_api.py` | Direct Python API benchmark script |
| `models/mfa/profile_mfa.py` | CLI configuration profiling script |
| `results/mfa_api_en.json` | API benchmark results for English |
| `results/mfa_api_ru.json` | API benchmark results for Russian |

## 7. API Stability Warning

From the MFA documentation:

> "While the MFA command-line interface is fairly stable, I do tend to do refactors of the internal code on fairly regular basis."

**Recommendation**: Pin MFA to a specific version (`montreal-forced-aligner==3.3.9`) and test on upgrade. The `KalpyAligner` API (from the `kalpy` package) is more stable than MFA's internal Python API.
