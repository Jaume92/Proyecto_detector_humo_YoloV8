from ultralytics import YOLO
import os
import shutil

# ================= CONFIG =================

MODEL_PATH = "yolov8m.pt"

TRAIN_INPUT = "data/raw/train/images"
VAL_INPUT = "data/raw/valid/images"

OUT_TRAIN_IMG = "data/processed/train/images"
OUT_TRAIN_LBL = "data/processed/train/labels"
OUT_VAL_IMG = "data/processed/val/images"
OUT_VAL_LBL = "data/processed/val/labels"

CONF_THRESHOLD = 0.4

MAX_IMAGES = 100   # <<< TEST LOCAL (en cloud poner None)

# =========================================

os.makedirs(OUT_TRAIN_IMG, exist_ok=True)
os.makedirs(OUT_TRAIN_LBL, exist_ok=True)
os.makedirs(OUT_VAL_IMG, exist_ok=True)
os.makedirs(OUT_VAL_LBL, exist_ok=True)

print("🚀 Loading YOLO model...")
model = YOLO(MODEL_PATH)


def process_folder(input_folder, out_img, out_lbl, limit=None):

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]

    if limit:
        images = images[:limit]

    print(f"📸 Processing {len(images)} images from {input_folder}")

    saved = 0

    for img_name in images:

        img_path = os.path.join(input_folder, img_name)

        results = model(img_path, conf=CONF_THRESHOLD, verbose=False)

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            continue

        label_path = os.path.join(out_lbl, img_name.replace(".jpg", ".txt"))

        with open(label_path, "w") as f:
            for box in boxes:
                cls = 0   # FORCE SMOKE CLASS
                x, y, w, h = box.xywhn[0]

                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        shutil.copy(img_path, os.path.join(out_img, img_name))
        saved += 1

    print(f"✅ Saved {saved} labeled images\n")


print("🔥 Processing TRAIN...")
process_folder(TRAIN_INPUT, OUT_TRAIN_IMG, OUT_TRAIN_LBL, MAX_IMAGES)

print("🔥 Processing VAL...")
process_folder(VAL_INPUT, OUT_VAL_IMG, OUT_VAL_LBL, MAX_IMAGES)

print("🎉 Pseudo-labeling finished")
