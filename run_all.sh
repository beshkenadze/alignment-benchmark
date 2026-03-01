#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================="
echo " Forced Alignment Benchmark Runner"
echo "============================================="
echo ""

MODELS=(stable_ts whisperx ctc_aligner faster_whisper qwen3_fa)

for model in "${MODELS[@]}"; do
    model_dir="$SCRIPT_DIR/models/$model"
    
    if [ ! -f "$model_dir/bench.py" ]; then
        echo "⚠️  Skipping $model — bench.py not found"
        continue
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ Running: $model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Setup venv if not exists
    if [ ! -d "$model_dir/.venv" ]; then
        echo "  Setting up venv..."
        if [ -f "$model_dir/setup.sh" ]; then
            bash "$model_dir/setup.sh"
        fi
    fi
    
    # Run benchmark
    if [ -d "$model_dir/.venv" ]; then
        "$model_dir/.venv/bin/python" "$model_dir/bench.py" \
            --config "$SCRIPT_DIR/config.json" \
            --output-dir "$SCRIPT_DIR/results"
    else
        echo "  ❌ No venv found for $model — run setup.sh first"
    fi
    
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶ Comparing results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 "$SCRIPT_DIR/compare.py"
