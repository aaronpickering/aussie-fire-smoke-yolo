import os
import numpy as np
from PIL import Image
from scipy import ndimage
import shutil

BASE = "/data/aussmoke_src"
OUT = "/data/merged"

MIN_AREA_PX = 100  # ignore tiny noise specks

def mask_to_boxes(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    h, w = mask.shape
    binary = mask > 127
    labeled, num_features = ndimage.label(binary)
    boxes = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < MIN_AREA_PX:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        box_w = (x_max - x_min) / w
        box_h = (y_max - y_min) / h
        boxes.append((x_center, y_center, box_w, box_h))
    return boxes

def process_split(src_split, target_split):
    img_dir = os.path.join(BASE, src_split, "AuSmoke", "images")
    mask_dir = os.path.join(BASE, src_split, "AuSmoke", "masks")
    out_img_dir = os.path.join(OUT, target_split, "images")
    out_lbl_dir = os.path.join(OUT, target_split, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    count = 0
    total_boxes = 0
    files = sorted(os.listdir(img_dir))
    for fname in files:
        stem = os.path.splitext(fname)[0]
        mask_path = os.path.join(mask_dir, stem + ".png")
        if not os.path.exists(mask_path):
            continue
        boxes = mask_to_boxes(mask_path)
        if not boxes:
            continue  # skip images with no significant smoke region

        new_name = f"aussmoke_{src_split.lower()}_{fname}"
        new_lbl_name = f"aussmoke_{src_split.lower()}_{stem}.txt"

        shutil.copy2(os.path.join(img_dir, fname), os.path.join(out_img_dir, new_name))
        with open(os.path.join(out_lbl_dir, new_lbl_name), "w") as f:
            for (xc, yc, bw, bh) in boxes:
                f.write(f"1 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")  # class 1 = smoke

        count += 1
        total_boxes += len(boxes)
        if count % 2000 == 0:
            print(f"  {src_split}: {count} done")

    print(f"{src_split} -> {target_split}: {count} images, {total_boxes} boxes")

process_split("Train", "train")
process_split("Test", "valid")
print("AusSmoke conversion complete.")
