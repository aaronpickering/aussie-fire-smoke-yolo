#!/bin/bash
set -e

RUN_NAME=${1:-fire_smoke_v2}
RUN_DIR="/mnt/bulk/fire-smoke-training/runs/$RUN_NAME"
REPO_DIR="/home/aaronp/aussie-fire-smoke-yolo"
RUNS_DIR="/mnt/bulk/fire-smoke-training/runs"

if [ ! -f "$RUN_DIR/weights/best.pt" ]; then
    echo "Error: no best.pt found at $RUN_DIR/weights/best.pt"
    exit 1
fi

echo "Copying model weights from $RUN_NAME..."
cp "$RUN_DIR/weights/best.pt" "$REPO_DIR/models/aussie-fire-smoke-yolov8s.pt"

echo "Updating README metrics from results.csv..."
python3 "$REPO_DIR/scripts/update_readme_metrics.py" "$RUN_DIR/results.csv" "$REPO_DIR/README.md"

echo "Regenerating training chart..."
docker run --rm \
  -v "$REPO_DIR:/repo" \
  -v "$RUNS_DIR:/runs" \
  python:3.11-slim bash -c "pip install --quiet pandas matplotlib && python3 /repo/scripts/generate_training_chart.py /runs/$RUN_NAME/results.csv /repo/assets/training_curves.png 'Fire/Smoke YOLOv8s -- run: $RUN_NAME'"

cd "$REPO_DIR"
git add .
git commit -m "Update model, metrics, and training chart from training run: $RUN_NAME"
git push origin main

echo "Done. Pushed updated model, metrics, and chart from $RUN_NAME."
