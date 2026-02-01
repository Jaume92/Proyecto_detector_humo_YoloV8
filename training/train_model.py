from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Device:", device)

model = YOLO("yolov8s.pt")   # s = sweet spot calidad/velocidad

model.train(
    data="data/fire_smoke.yaml",
    epochs=80,
    imgsz=640,
    batch=8,
    patience=15,
    device=device,
    name="fire_smoke_v1",
    workers=2,
    optimizer="AdamW",
    lr0=0.003,
    verbose=True
)

print("✅ Training finished")
