from ultralytics import YOLO
import cv2

# ================= CONFIG =================

MODEL_PATH = "models/weights/best.pt"
CAMERA_INDEX = 0         # prueba 0 o 1 según Iriun
CONF_THRESHOLD = 0.45

# ================= LOAD MODEL =================

print("🚀 Loading model...")
model = YOLO(MODEL_PATH)

# ================= OPEN CAMERA =================

print("📷 Opening camera...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not found")
    exit()

print("✅ Camera ready. Press Q to quit")

# ================= MAIN LOOP =================

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame not received")
        break

    # INFERENCE (solo smoke y fire)
    results = model(frame, conf=CONF_THRESHOLD, classes=[0, 1])

    annotated_frame = results[0].plot()

    cv2.imshow("🔥 Fire & Smoke Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= CLEANUP =================

cap.release()
cv2.destroyAllWindows()
print("👋 Closed")
