# Forced Alignment Benchmark

Benchmark of 9 word-level timestamp models for speech alignment on Apple Silicon (M1 Max, 32GB).

Tested on two audio samples:
- **English** (30s) — US Contract Law lecture
- **Russian** (10s) — podcast excerpt

## Results

### English — Cross-model deviation from consensus mean (lower = better)

| # | Model | Mean Start Dev | Median Start | Inference | License |
|---|---|---:|---:|---:|---|
| 🥇 | **MFA v3** | **69.3ms** | **39.0ms** | 35.1s (CLI) / **1.49s (API)** | MIT |
| 🥈 | **ctc-forced-aligner (MMS)** | 73.3ms | 31.2ms | 4.6s | MIT |
| 3 | qwen3-forced-aligner | 86.1ms | 52.7ms | 2.7s | Apache 2.0 |
| 4 | whisper-char-align (large-v3) | 94.6ms | 55.9ms | 20.1s | MIT |
| 5 | parakeet-tdt-0.6b-v3 (mlx) | 98.0ms | 70.9ms | 3.4s | CC-BY-4.0 |
| 6 | stable-ts (base) | 102.4ms | 58.9ms | 4.0s | MIT |
| 7 | vosk (small-en) | 108.8ms | 30.8ms | 2.7s | Apache 2.0 |
| 8 | whisperx | 126.3ms | 79.6ms | 1.1s | MIT |
| 9 | faster-whisper | 147.3ms | 55.7ms | 2.7s | MIT |

### Russian — Cross-model deviation from consensus mean (lower = better)

| # | Model | Mean Start Dev | Median Start | Inference | License |
|---|---|---:|---:|---:|---|
| 🥇 | **Vosk (ru-0.42)** | **45.1ms** | **31.2ms** | 1.0s | Apache 2.0 |
| 🥈 | **MFA v3 (russian_mfa)** | **56.7ms** | 41.4ms | 190.8s (CLI) / **1.61s (API)** | MIT |
| 3 | qwen3-forced-aligner | 63.2ms | 41.6ms | 0.25s | Apache 2.0 |
| 4 | stable-ts (base) | 74.6ms | 57.8ms | 0.6s | MIT |
| 5 | parakeet-tdt-0.6b-v3 (mlx) | 79.1ms | 69.1ms | 0.6s | CC-BY-4.0 |
| 6 | whisperx | 82.1ms | 48.4ms | 0.8s | MIT |
| 7 | whisper-char-align (large-v3) | 86.5ms | 77.8ms | 4.2s | MIT |
| 8 | faster-whisper | 170.6ms | 65.8ms | 1.1s | MIT |
| 9 | ctc-forced-aligner (MMS) | 207.1ms | 262.8ms | 0.75s | MIT |

## Key Findings

- **MFA v3** is the gold standard for accuracy (69ms EN, 57ms RU). The CLI is slow (35–191s), but using the direct `KalpyAligner` Python API achieves **1.5s per utterance** with negligible accuracy loss. See [MFA_ANALYSIS.md](MFA_ANALYSIS.md) for the deep-dive.
- **Vosk** is a surprise winner for Russian (45ms mean deviation), very fast, Apache 2.0 — but it's an ASR model that ignores the provided transcript
- **Qwen3-ForcedAligner** offers the best speed/accuracy trade-off for both languages (3rd place EN and RU, fastest among accurate models)
- **0% cross-model agreement** within 50ms for either language — forced alignment has inherent ~50–100ms uncertainty
- All models were benchmarked using only permissive open-source licenses (MIT, Apache 2.0, CC-BY-4.0)

## Models Tested

| Model | Type | Languages | Install |
|---|---|---|---|
| [stable-ts](https://github.com/jianfch/stable-ts) | Whisper DTW alignment | 99 | `pip install stable-ts` |
| [WhisperX](https://github.com/m-bain/whisperX) | Whisper + wav2vec2 alignment | 99 | `pip install whisperx` |
| [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) | MMS CTC alignment | 1100+ | `pip install ctc-forced-aligner` |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2 Whisper | 99 | `pip install faster-whisper` |
| [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | LLM NAR alignment | 11 | `pip install mlx-audio` |
| [Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | TDT native timestamps | 25 EU | `pip install parakeet-mlx` |
| [Vosk](https://alphacephei.com/vosk/) | Kaldi ASR + timestamps | 20+ | `pip install vosk` |
| [MFA v3](https://montreal-forced-aligner.readthedocs.io/) | GMM-HMM forced alignment | ~30 | `conda install montreal-forced-aligner` |
| [whisper-char-align](https://github.com/30stomercury/whisper-char-alignment) | Character-level DTW | 99 | via stable-ts `aligner="new"` |

## How to Run

Each model has its own directory under `models/` with:
- `setup.sh` — creates a virtual environment and installs dependencies
- `bench.py` — runs the benchmark and outputs JSON results

```bash
# Example: run one model
cd models/stable_ts
bash setup.sh
.venv/bin/python bench.py --config ../../config.json --output-dir ../../results

# Run all models
bash run_all.sh

# Compare results
python3 compare.py
```

## Hardware

- Apple Silicon M1 Max, 32GB RAM
- macOS, Python 3.12 via uv
- CPU inference only (no CUDA), MLX where available

## Methodology

Cross-model consensus comparison: for each word, compute the mean timestamp across all models, then measure each model's deviation from that mean. This avoids the need for ground-truth annotations (which don't exist for these audio files).

Limitations:
- Only 2 audio samples (1 EN, 1 RU) — not statistically robust
- No ground-truth word boundaries — relative comparison only
- Models that recognize fewer words get fewer comparison points
