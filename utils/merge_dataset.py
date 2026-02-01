import os
import shutil

SMOKE_TRAIN_IMG = "data/processed/train/images"
SMOKE_TRAIN_LBL = "data/processed/train/labels"
SMOKE_VAL_IMG = "data/processed/val/images"
SMOKE_VAL_LBL = "data/processed/val/labels"

FIRE_TRAIN_IMG = "data/raw/fire/train/images"
FIRE_TRAIN_LBL = "data/raw/fire/train/labels"
FIRE_VAL_IMG = "data/raw/fire/valid/images"
FIRE_VAL_LBL = "data/raw/fire/valid/labels"

OUT_TRAIN_IMG = "data/final/train/images"
OUT_TRAIN_LBL = "data/final/train/labels"
OUT_VAL_IMG = "data/final/val/images"
OUT_VAL_LBL = "data/final/val/labels"

os.makedirs(OUT_TRAIN_IMG, exist_ok=True)
os.makedirs(OUT_TRAIN_LBL, exist_ok=True)
os.makedirs(OUT_VAL_IMG, exist_ok=True)
os.makedirs(OUT_VAL_LBL, exist_ok=True)

def copy_set(img_src, lbl_src, img_dst, lbl_dst, tag):

    images = [f for f in os.listdir(img_src) if f.endswith(".jpg")]

    copied = 0

    for img in images:

        src_img = os.path.join(img_src, img)
        dst_img = os.path.join(img_dst, img)

        lbl_name = img.replace(".jpg", ".txt")
        src_lbl = os.path.join(lbl_src, lbl_name)
        dst_lbl = os.path.join(lbl_dst, lbl_name)

        if not os.path.exists(src_lbl):
            continue

        shutil.copy(src_img, dst_img)
        shutil.copy(src_lbl, dst_lbl)

        copied += 1

    print(f"✅ {tag}: copied {copied} samples")


print("🔥 Merging SMOKE TRAIN")
copy_set(SMOKE_TRAIN_IMG, SMOKE_TRAIN_LBL, OUT_TRAIN_IMG, OUT_TRAIN_LBL, "Smoke train")

print("🔥 Merging FIRE TRAIN")
copy_set(FIRE_TRAIN_IMG, FIRE_TRAIN_LBL, OUT_TRAIN_IMG, OUT_TRAIN_LBL, "Fire train")

print("🔥 Merging SMOKE VAL")
copy_set(SMOKE_VAL_IMG, SMOKE_VAL_LBL, OUT_VAL_IMG, OUT_VAL_LBL, "Smoke val")

print("🔥 Merging FIRE VAL")
copy_set(FIRE_VAL_IMG, FIRE_VAL_LBL, OUT_VAL_IMG, OUT_VAL_LBL, "Fire val")

print("🎉 Dataset merged successfully")
