#!/bin/bash
set -e

RUN_NAME=${1:-fire_smoke_v2}
RUN_DIR="/mnt/bulk/fire-smoke-training/runs/$RUN_NAME"
REPO_DIR="/home/aaronp/aussie-fire-smoke-yolo"

if [ ! -f "$RUN_DIR/weights/best.pt" ]; then
    echo "Error: no best.pt found at $RUN_DIR/weights/best.pt"
    exit 1
fi

echo "Copying model weights from $RUN_NAME..."
cp "$RUN_DIR/weights/best.pt" "$REPO_DIR/models/aussie-fire-smoke-yolov8s.pt"

echo "Updating README metrics from results.csv..."
python3 "$REPO_DIR/scripts/update_readme_metrics.py" "$RUN_DIR/results.csv" "$REPO_DIR/README.md"

echo "Regenerating training chart..."
python3 "$REPO_DIR/scripts/generate_training_chart.py" "$RUN_DIR/results.csv" "$REPO_DIR/assets/training_curves.png" "Fire/Smoke YOLOv8s -- run: $RUN_NAME"

cd "$REPO_DIR"
git add .
git commit -m "Update model and metrics from training run: $RUN_NAME"
git push origin main

echo "Done. Pushed updated model and metrics from $RUN_NAME."
