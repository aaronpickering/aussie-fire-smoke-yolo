import csv
import sys

results_csv = sys.argv[1]
readme_path = sys.argv[2]

with open(results_csv) as f:
    reader = csv.DictReader(f)
    rows = list(reader)
last = rows[-1]

precision = float(last['metrics/precision(B)'])
recall = float(last['metrics/recall(B)'])
map50 = float(last['metrics/mAP50(B)'])
map50_95 = float(last['metrics/mAP50-95(B)'])
epoch = last['epoch']

new_block = f"""### Validation performance
<!-- METRICS_START -->
| Metric | Overall |
|---|---|
| Precision | {precision:.3f} |
| Recall | {recall:.3f} |
| mAP50 | {map50:.3f} |
| mAP50-95 | {map50_95:.3f} |

(Metrics after {epoch} epochs on the full combined dataset. Per-class
fire/smoke breakdown is available in the training logs.)
<!-- METRICS_END -->"""

content = open(readme_path).read()
start_marker = "### Validation performance"
end_marker = "<!-- METRICS_END -->"
start_idx = content.index(start_marker)
end_idx = content.index(end_marker) + len(end_marker)
content = content[:start_idx] + new_block + content[end_idx:]
open(readme_path, 'w').write(content)
print(f"README updated: P={precision:.3f} R={recall:.3f} mAP50={map50:.3f} mAP50-95={map50_95:.3f}")
