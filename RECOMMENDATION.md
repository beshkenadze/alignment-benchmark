# Recommendation: Language-Routed Forced Alignment for Solovey

## Architecture

Since Solovey is a text-based audio editor, the transcript is already available. This means we should use **forced aligners** (audio + transcript → word timestamps), not ASR models that ignore the transcript.

### Primary: MFA v3 via Python API (both languages)

After deep-dive analysis ([MFA_ANALYSIS.md](MFA_ANALYSIS.md)), MFA's CLI slowness (35–191s) turned out to be **framework overhead, not alignment speed**. Using the direct `KalpyAligner` Python API, MFA becomes fast enough for interactive use:

```
MFA Alignment Service (long-running process)
  │
  │  Startup (once):
  │    • Load acoustic models     ~1s
  │    • Load cached lexicon FSTs  ~0.1s (pre-compiled)
  │
  │  Per-request (~1.5s):
  │    1. MFCC extraction          ~1.3s
  │    2. Viterbi alignment         ~0.1–0.3s
  │
  Language Router:
    ├─ EN → MFA english_mfa       96ms mean dev, 40ms median, 1.49s
    │       MIT, conda install montreal-forced-aligner
    │
    ├─ RU → MFA russian_mfa       55ms mean dev, 46ms median, 1.61s
    │       MIT, conda install montreal-forced-aligner
    │
    └─ Fallback (unsupported lang) → qwen3-forced-aligner
            Supports 11 languages, Apache 2.0
```

### Fallback: qwen3-forced-aligner

For languages without an MFA acoustic model, use qwen3-forced-aligner (0.6B). It supports 11 languages and is the fastest among accurate models (0.25s for 10s audio).

## Why MFA via API

### The breakthrough

MFA's CLI (`mfa align`) is slow because of framework overhead — database startup, corpus ingestion, FST compilation, speaker adaptation, multiprocessing spawn. **The actual Kaldi alignment takes 0.1–0.3s.** By calling `KalpyAligner` directly via Python, we skip all overhead:

| Metric | CLI (`mfa align`) | Python API | Speedup |
|--------|-------------------|------------|---------|
| EN (30s audio) | 35.1s | 1.49s | 23× |
| RU (10s audio) | 190.8s | 1.61s | 118× |
| EN accuracy | 67.6ms | 96.3ms | −29ms (no speaker adaptation) |
| RU accuracy | 51.4ms | 55.3ms | −4ms (negligible) |

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

## Production Integration

```python
# Solovey integration — keep model loaded, align per-request
from kalpy.aligner import KalpyAligner
from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.utterance import Segment, Utterance as KalpyUtterance
from montreal_forced_aligner.models import AcousticModel

# Load ONCE at startup:
acoustic_model = AcousticModel("english_mfa.zip")
lexicon_compiler = LexiconCompiler(...)
lexicon_compiler.load_pronunciations("english_mfa.dict")
lexicon_compiler.create_fsts()   # cache to disk for fast reload
kalpy_aligner = KalpyAligner(acoustic_model, lexicon_compiler, beam=10)

# Per-request (~1.5s):
seg = Segment("audio.wav", 0, None, 0)
utt = KalpyUtterance(seg, transcript.lower())
utt.generate_mfccs(acoustic_model.mfcc_computer)
cmvn = CmvnComputer().compute_cmvn_from_features([utt.mfccs])
utt.apply_cmvn(cmvn)
ctm = kalpy_aligner.align_utterance(utt)
for wi in ctm.word_intervals:
    print(f"{wi.label}: {wi.begin:.3f}–{wi.end:.3f}s")
```

See [bench_api.py](models/mfa/bench_api.py) for the complete working implementation.

## Models NOT Recommended

- **Vosk / Parakeet-TDT** — good ASR models, but useless as forced aligners (they ignore the provided transcript)
- **whisper-char-align** — didn't match paper claims, slow due to large-v3 model
- **MFA CLI** (`mfa align`) — use the Python API instead; CLI overhead makes it 23–118× slower than necessary

## Forced Aligner vs ASR Classification

| Model | Forced Aligner? | Uses provided transcript |
|---|---|---|
| MFA v3 (API) | ✅ | yes |
| ctc-forced-aligner | ✅ | yes |
| qwen3-forced-aligner | ✅ | yes |
| stable-ts (align) | ✅ | yes |
| whisper-char-align | ✅ | yes |
| whisperx | ✅ | yes (alignment step) |
| Vosk | ❌ ASR | no — does its own recognition |
| Parakeet-TDT | ❌ ASR | no — does its own recognition |
| faster-whisper | ❌ ASR | no — does its own recognition |
