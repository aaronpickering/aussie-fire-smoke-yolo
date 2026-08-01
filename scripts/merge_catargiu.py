import os
import shutil
from pathlib import Path

BULK = Path("/mnt/bulk/fire-smoke-training")
OUT = BULK / "merged"

SOURCES = [
    (BULK / "catargiu/part1/FireSmokeDataset", "catargiu_p1"),
    (BULK / "catargiu/part2/FireSmokeNEWdataset", "catargiu_p2"),
]

# Source scheme: 0=fire, 1=other (drop), 2=smoke -> remap to 0=fire, 1=smoke
REMAP = {"0": "0", "2": "1"}  # "1" (other) intentionally absent -> dropped

SPLIT_MAP = {"train": "train", "valid": "valid", "test": "test"}

total_images = 0
total_boxes_kept = 0
total_boxes_dropped = 0

for src_root, prefix in SOURCES:
    for src_split, target_split in SPLIT_MAP.items():
        img_dir = src_root / src_split / "images"
        lbl_dir = src_root / src_split / "labels"
        if not img_dir.exists():
            print(f"SKIP (missing): {img_dir}")
            continue

        out_img_dir = OUT / target_split / "images"
        out_lbl_dir = OUT / target_split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"

            new_name = f"{prefix}_{img_path.name}"
            new_lbl_name = f"{prefix}_{stem}.txt"

            shutil.copy2(img_path, out_img_dir / new_name)
            total_images += 1

            new_lines = []
            if lbl_path.exists():
                for line in lbl_path.read_text().strip().splitlines():
                    if not line.strip():
                        continue
                    parts = line.split()
                    src_class = parts[0]
                    if src_class in REMAP:
                        parts[0] = REMAP[src_class]
                        new_lines.append(" ".join(parts))
                        total_boxes_kept += 1
                    else:
                        total_boxes_dropped += 1  # "other" class, dropped

            (out_lbl_dir / new_lbl_name).write_text("\n".join(new_lines))
            count += 1

        print(f"{prefix}/{src_split}: {count} images -> {target_split}")

print(f"\nTOTAL: {total_images} images, {total_boxes_kept} boxes kept, {total_boxes_dropped} 'other' boxes dropped")
