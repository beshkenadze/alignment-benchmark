# Recommendation: Language-Routed Forced Alignment for Solovey

## Architecture

Since Solovey is a text-based audio editor, the transcript is already available. This means we should use **forced aligners** (audio + transcript → word timestamps), not ASR models that ignore the transcript.

### Primary: SwiftKaldiAligner (native, both languages)

After the MFA Python API deep-dive ([MFA_ANALYSIS.md](MFA_ANALYSIS.md)), we went further and reimplemented the entire Kaldi forced alignment pipeline in C++ with a Swift wrapper: **[SwiftKaldiAligner](https://github.com/beshkenadze/SwiftKaldiAligner)**. Zero Python runtime dependency.

```
SwiftKaldiAligner (SPM package)
  │
  │  Model load (once):
  │    • EN: 0.23s  /  RU: 1.28s
  │
  │  Per-request:
  │    • EN (30s audio): 0.13s  —  79 words
  │    • RU (10s audio): 0.11s  —  25 words
  │
  Language Router:
    ├─ EN → english_mfa model      ~100ms mean dev, 0.13s
    │       MIT, SPM dependency
    │
    ├─ RU → russian_mfa model      ~100ms mean dev, 0.11s
    │       MIT, SPM dependency
    │
    └─ Fallback (unsupported lang) → qwen3-forced-aligner
            Supports 11 languages, Apache 2.0
```

### Fallback: qwen3-forced-aligner

For languages without an MFA acoustic model, use qwen3-forced-aligner (0.6B). It supports 11 languages and is the fastest among accurate models (0.25s for 10s audio).

## Why SwiftKaldiAligner

### The progression

| Implementation | EN (30s) | RU (10s) | Python required |
|---|---|---|---|
| MFA CLI (`mfa align`) | 35.1s | 190.8s | yes + conda |
| MFA Python API (`KalpyAligner`) | 1.49s | 1.61s | yes |
| **SwiftKaldiAligner** | **0.13s** | **0.11s** | **no** |

### Swift vs Python Kaldi alignment (detailed)

#### Short clips (standalone Kaldi benchmark)

| Metric | Swift EN (30s) | Python EN (30s) | Swift advantage | Swift RU (10s) | Python RU (10s) | Swift advantage |
|---|---|---|---|---|---|---|
| **Model load** | 0.23s | 4.67s | **20x** | 1.28s | 26.41s | **21x** |
| &nbsp;&nbsp;acoustic model | instant | 0.52s | | instant | 0.55s | |
| &nbsp;&nbsp;lexicon FST | instant | 1.63s | | instant | 22.83s | |
| &nbsp;&nbsp;Python import | — | 2.16s | | — | 2.37s | |
| **Inference** | 0.13s | 1.49s | **11x** | 0.11s | 1.61s | **15x** |
| &nbsp;&nbsp;features (MFCC+LDA) | ~0.05s | 1.37s | | ~0.04s | 1.27s | |
| &nbsp;&nbsp;Viterbi decode | ~0.08s | 0.12s | | ~0.07s | 0.34s | |
| Words aligned | 79 | 66 (+14 unk) | | 25 | 20 (+5 unk) | |

#### Full pipeline (19-min real audio)

| Metric | Swift EN | Python EN | Swift | Swift RU | Python RU | Swift |
|---|---|---|---|---|---|---|
| **Align model load** | 0.37s | 2.56s | **6.9x** | 1.35s | 28.77s | **21x** |
| **Align inference** | 4.28s | 7.39s | **1.7x** | 3.74s | 52.91s | **14x** |
| Words aligned | 2886 | 2933 | | 2429 | 2469 | |
| Segments | 163 | 287 | | 112 | 166 | |
| Failures | 0 | 0 | | 0 | 0 | |

**Why Python RU alignment is so slow**: `kalpy` rebuilds the lexicon FST from the 452K-word Russian dictionary on every model load (22.8s). Swift pre-compiles the FST once during C++ initialization. For inference, Python's per-utterance CMVN and temp file I/O add overhead that scales with segment count.

SwiftKaldiAligner wraps Kaldi/OpenFst C++ static libraries via a C-style opaque handle API (`extern "C"`), exposed to Swift through SPM. Same acoustic models, same dictionary files, same MFCC→CMVN→Splice→LDA→HCLG→Viterbi pipeline — just without Python overhead.

### Accuracy parity

Word timings match MFA Python API within ~100ms (well within the inherent ~50–100ms uncertainty of forced alignment). SwiftKaldiAligner finds more words than MFA Python (79 vs 66+14 `<unk>` for EN, 25 vs 20+5 `<unk>` for RU) because it uses full dictionary pronunciations instead of mapping unknowns to `spn`.

### Why not ctc-forced-aligner for EN?

ctc-forced-aligner (73ms EN) is slightly more accurate than MFA API (96ms EN) for English, but:
- ctc-aligner is **catastrophic for Russian** (207ms mean, 263ms median — last place!)
- MFA handles both languages well with a single architecture
- MFA API median (40ms) matches ctc-aligner median (32ms) — the mean difference is driven by outliers
- Simpler architecture: one model service vs two different alignment engines

### Why not qwen3-forced-aligner for everything?

Qwen3 (87ms EN, 58ms RU) is a strong universal option but MFA API is more accurate for both languages while being similarly fast (~1.5s vs ~0.25–2.7s).

## Safety Margin

**±75ms padding** on word boundaries for audio splicing — based on median deviations of best models (40–46ms) with safety buffer.

## Full Pipeline Benchmark

End-to-end tests on real ~19-minute audio files.

### Russian (19-min podcast, 1143s)

| Stage | Swift | Python | Swift advantage |
|---|---|---|---|
| Audio load | 2.12s | 2.00s | — |
| VAD | 5.93s | 5.47s | — |
| ASR model load | 1.84s | 2.51s | 1.4x |
| **ASR inference** | **67.12s** | **134.57s** | **2.0x** |
| Alignment model load | 1.35s | 28.77s | **21.3x** |
| **Alignment inference** | **3.74s** | **52.91s** | **14.1x** |
| **TOTAL** | **82.1s** | **226.2s** | **2.75x** |
| RTFx | 13.9x | 5.1x | |

### English (19-min audiobook chapter, 1163s)

| Stage | Swift | Python | Swift advantage |
|---|---|---|---|
| Audio load | 0.73s | 1.64s | 2.2x |
| VAD | 1.44s | 5.14s | 3.6x |
| ASR model load | 2.13s | 2.71s | 1.3x |
| **ASR inference** | **51.86s** | **284.42s** | **5.5x** |
| Alignment model load | 0.37s | 2.56s | 6.9x |
| **Alignment inference** | **4.28s** | **7.39s** | **1.7x** |
| **TOTAL** | **60.8s** | **303.9s** | **5.0x** |
| RTFx | 19.1x | 3.8x | |

### Pipeline details

- **Swift**: Audio Load → VAD (Silero/CoreML) → ASR (Qwen3-ASR 0.6B-4bit / MLX GPU) → Alignment (Kaldi C++ / CPU)
- **Python**: Audio Load → VAD (Silero/PyTorch CPU) → ASR (Whisper-large-v3-turbo / MLX GPU) → Alignment (Kaldi/kalpy CPU)
- **Note**: Python uses Whisper-large-v3-turbo (not Qwen3-ASR) because mlx-audio Python 0.3.1 doesn't include Qwen3-ASR yet

### Key observations

1. **Swift is 2.75–5.0x faster end-to-end** depending on language
2. **English is faster** for both pipelines — smaller dictionary (42K vs 452K words), cleaner audio (studio recording vs podcast)
3. **ASR dominates** both pipelines (85–94% of total time) — alignment itself is cheap
4. **Alignment biggest win on Russian**: Swift 14x faster (3.7s vs 52.9s) due to Python's heavy model load + per-utterance CMVN overhead
5. **ASR biggest win on English**: Swift Qwen3-ASR 5.5x faster than Python Whisper — English benefits most from the smaller model
6. **Zero alignment failures** across all runs (both languages, both pipelines)

## Production Integration

### Swift (recommended for iOS/macOS)

```swift
import SwiftKaldiAligner

// Load ONCE at startup:
let aligner = try SwiftKaldiAligner(
    modelDir: "/path/to/english_mfa",
    dictPath: "/path/to/english_mfa.dict"
)

// Per-request (~0.13s for 30s audio):
let words = try aligner.align(
    audio: pcmSamples,
    sampleRate: 16000,
    transcript: "hello world"
)
for w in words {
    print("\(w.word): \(w.startTime)–\(w.endTime)s")
}
```

See [SwiftKaldiAligner](https://github.com/beshkenadze/SwiftKaldiAligner) for full setup instructions.

### Python fallback (for unsupported languages)

```python
# For languages without MFA models, use qwen3-forced-aligner
from mlx_audio.tts import generate_alignment
```

## Models NOT Recommended

- **Vosk / Parakeet-TDT** — good ASR models, but useless as forced aligners (they ignore the provided transcript)
- **whisper-char-align** — didn't match paper claims, slow due to large-v3 model
- **MFA CLI** (`mfa align`) — use the Python API instead; CLI overhead makes it 23–118× slower than necessary

## Forced Aligner vs ASR Classification

| Model | Forced Aligner? | Uses provided transcript | Runtime |
|---|---|---|---|
| SwiftKaldiAligner | ✅ | yes | Swift (native) |
| MFA v3 (API) | ✅ | yes | Python |
| ctc-forced-aligner | ✅ | yes | Python |
| qwen3-forced-aligner | ✅ | yes | Python |
| stable-ts (align) | ✅ | yes | Python |
| whisper-char-align | ✅ | yes | Python |
| whisperx | ✅ | yes (alignment step) | Python |
| Vosk | ❌ ASR | no — does its own recognition | Python |
| Parakeet-TDT | ❌ ASR | no — does its own recognition | Python |
| faster-whisper | ❌ ASR | no — does its own recognition | Python |
