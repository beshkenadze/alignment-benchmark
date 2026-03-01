# Recommendation: Language-Routed Forced Alignment for Solovey

## Architecture

Since Solovey is a text-based audio editor, the transcript is already available. This means we should use **forced aligners** (audio + transcript → word timestamps), not ASR models that ignore the transcript.

```
Language Detector
    ├─ EN → ctc-forced-aligner (MMS-300M)
    │       73ms mean dev, 31ms median, 4.6s/30s audio
    │       MIT, pip install ctc-forced-aligner
    │
    ├─ RU → qwen3-forced-aligner (0.6B)
    │       63ms mean dev, 42ms median, 0.25s/10s audio
    │       Apache 2.0, pip install mlx-audio
    │
    ├─ FR/DE/ES/IT/PT/JA/KO/ZH → qwen3-forced-aligner
    │       Supports 11 languages, fastest model
    │
    └─ Other EU languages → ctc-forced-aligner (MMS)
            1100+ languages via MMS, universal fallback
```

## Why These Models

### EN → ctc-forced-aligner (not qwen3)
- More accurate for EN (73ms vs 86ms mean, **31ms vs 53ms median**)
- True CTC alignment on MMS — designed for alignment
- Lightweight (300M params)

### RU → qwen3-forced-aligner (not ctc)
- ctc-aligner is catastrophic for RU (207ms mean, 263ms median — last place!)
- Qwen3 is 3rd for RU (63ms) and **18x faster** than MFA with comparable accuracy
- MFA is more accurate (57ms) but 191 seconds for 10s audio — unacceptable for interactive editing

## Safety Margin

**±75ms padding** on word boundaries for audio splicing — based on median deviations of best models (31-42ms) with safety buffer.

## Models NOT Recommended

- **MFA v3** — gold standard accuracy, but too slow for interactive use (190s for 10s RU audio). Only viable for offline batch processing.
- **Vosk / Parakeet-TDT** — good ASR models, but useless as aligners (they ignore the provided transcript)
- **whisper-char-align** — didn't match paper claims, slow due to large-v3 model

## Forced Aligner vs ASR Classification

| Model | Forced Aligner? | Uses provided transcript |
|---|---|---|
| ctc-forced-aligner | ✅ | yes |
| qwen3-forced-aligner | ✅ | yes |
| MFA v3 | ✅ | yes |
| stable-ts (align) | ✅ | yes |
| whisper-char-align | ✅ | yes |
| whisperx | ✅ | yes (alignment step) |
| Vosk | ❌ ASR | no — does its own recognition |
| Parakeet-TDT | ❌ ASR | no — does its own recognition |
| faster-whisper | ❌ ASR | no — does its own recognition |
