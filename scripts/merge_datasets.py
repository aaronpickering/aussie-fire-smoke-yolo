import os
import shutil
from pathlib import Path

BULK = Path("/mnt/bulk/fire-smoke-training")
OUT = BULK / "merged"

# (source_images_dir, source_labels_dir, remap_dict or None, target_split)
# remap_dict maps source_class_id (str) -> target_class_id (str)
# Target scheme: 0 = fire, 1 = smoke

SOURCES = []

# 1. Original CQU dataset (already fire=0, smoke=1) - train/valid/test
cqu = BULK / "data"
for split, target_split in [("train", "train"), ("valid", "valid"), ("test", "test")]:
    SOURCES.append((cqu / split / "images", cqu / split / "labels", None, target_split, f"cqu_{split}"))

# 2. D-Fire via Kaggle mirror (smoke=0, fire=1 -> needs swap) - data/train, data/val, data/test
dfire = BULK / "dfire_kaggle" / "data"
dfire_remap = {"0": "1", "1": "0"}  # swap smoke<->fire
for split, target_split in [("train", "train"), ("val", "valid"), ("test", "test")]:
    SOURCES.append((dfire / split / "images", dfire / split / "labels", dfire_remap, target_split, f"dfire_{split}"))

# 3. Kaggle hussainnasirkhan (already fire=0, smoke=1) - train/valid/test
hussain = BULK / "kaggle_hussain" / "Fire-and-Smoke-Detection-Dataset" / "Fire-and-Smoke-Detection-Dataset" / "dataset"
for split, target_split in [("train", "train"), ("valid", "valid"), ("test", "test")]:
    SOURCES.append((hussain / split / "images", hussain / split / "labels", None, target_split, f"hussain_{split}"))

# 4. Pyro-SDIS (already remapped to smoke=1 during export, smoke-only so no fire) - train/val
pyro = BULK / "pyro_sdis"
for split, target_split in [("train", "train"), ("val", "valid")]:
    SOURCES.append((pyro / "images" / split, pyro / "labels" / split, None, target_split, f"pyro_{split}"))

for split in ["train", "valid", "test"]:
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

total_images = 0
total_labels = 0

for img_dir, lbl_dir, remap, target_split, prefix in SOURCES:
    if not img_dir.exists():
        print(f"SKIP (missing): {img_dir}")
        continue
    img_files = list(img_dir.iterdir())
    count = 0
    for img_path in img_files:
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        new_name = f"{prefix}_{img_path.name}"
        new_lbl_name = f"{prefix}_{stem}.txt"

        dest_img = OUT / target_split / "images" / new_name
        dest_lbl = OUT / target_split / "labels" / new_lbl_name

        shutil.copy2(img_path, dest_img)
        total_images += 1

        if lbl_path.exists():
            lines = lbl_path.read_text().strip().splitlines()
            if remap:
                new_lines = []
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.split()
                    parts[0] = remap.get(parts[0], parts[0])
                    new_lines.append(" ".join(parts))
                dest_lbl.write_text("\n".join(new_lines))
            else:
                shutil.copy2(lbl_path, dest_lbl)
            total_labels += 1
        else:
            dest_lbl.write_text("")
            total_labels += 1

        count += 1
    print(f"{prefix}: {count} images copied ({target_split})")

print(f"\nTOTAL: {total_images} images, {total_labels} labels")

data_yaml = OUT / "data.yaml"
data_yaml.write_text(
    "train: train/images\n"
    "val: valid/images\n"
    "test: test/images\n\n"
    "nc: 2\n"
    "names: ['fire', 'smoke']\n"
)
print(f"Wrote {data_yaml}")
