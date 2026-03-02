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

## Full Pipeline Benchmark (19-min Russian podcast)

End-to-end test on a real 19-minute Russian podcast episode (1143s).

### Swift Pipeline (native)

```
Pipeline: Audio Load → VAD (Silero/CoreML) → ASR (Qwen3-ASR/MLX GPU) → Alignment (Kaldi/CPU)

Audio: 1143.1s (19 min), Russian
Segments: 112 speech segments (VAD)
Words: 2429 force-aligned
Total: 82.1s → 13.9x realtime
```

### Python Pipeline (comparison)

```
Pipeline: Audio Load → VAD (Silero/PyTorch CPU) → ASR (Whisper-large-v3-turbo/MLX GPU) → Alignment (Kaldi/kalpy CPU)

Audio: 1143.1s (19 min), Russian
Segments: 166 speech segments (VAD)
Words: 2469 force-aligned
Total: 226.2s → 5.1x realtime
```

### Swift vs Python comparison

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

### Key observations

1. **Swift is 2.75x faster end-to-end** — 82s vs 226s for the same 19-min podcast
2. **Alignment is the biggest win**: Swift Kaldi is **14x faster** than Python kalpy (3.7s vs 52.9s) — no Python interpreter, no temp file I/O, no per-utterance CMVN overhead
3. **Alignment model load**: Swift 1.35s vs Python 28.8s (**21x**) — Python loads MFA AcousticModel + builds LexiconCompiler from scratch; Swift pre-compiles everything in C++
4. **ASR**: Swift Qwen3-ASR is **2x faster** than Python Whisper-large-v3-turbo — smaller model (0.6B-4bit vs 809M) + native MLX-Swift vs Python MLX overhead
5. **VAD is comparable** — both use Silero, similar speed (~5.5s)
6. **Zero alignment failures** in both pipelines
7. **Note**: Python uses Whisper-large-v3-turbo (not Qwen3-ASR) because mlx-audio Python 0.3.1 doesn't include Qwen3-ASR yet

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
