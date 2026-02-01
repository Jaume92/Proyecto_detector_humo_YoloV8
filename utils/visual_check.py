from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "yolov8n.pt"   # usamos nano para ir rápido
IMAGE_FOLDER = "data/processed/train/images"

# Cargar modelo
model = YOLO(MODEL_PATH)

# Pillamos la primera imagen del processed
images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")]

if len(images) == 0:
    print("❌ No hay imágenes en processed/train/images")
    exit()

img_path = os.path.join(IMAGE_FOLDER, images[0])

print("📸 Visualizando:", img_path)

results = model(img_path)

annotated = results[0].plot()

cv2.imshow("Smoke Detection Check", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
