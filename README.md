Fire & Smoke Detection System (YOLOv8)

Sistema de detección de humo y fuego en tiempo real utilizando YOLOv8.
El proyecto implementa un pipeline completo de preparación de datos, pseudo-etiquetado, fusión de datasets y entrenamiento optimizado para GPU.

Características

Detección multi-clase: humo y fuego

Pipeline automático de pseudo-labeling para humo

Dataset unificado en formato YOLO

Entrenamiento preparado para GPU cloud

Estructura reproducible para producción

Estructura del proyecto
Proyecto_detector_humo_YoloV8/
│
├── data/
│   ├── raw/              # Datasets originales (no versionados)
│   └── fire_smoke.yaml   # Configuración YOLO final
│
├── utils/
│   ├── pseudo_label_smoke.py
│   ├── remap_fire_labels.py
│   └── merge_datasets.py
│
├── training/
│   └── train_model.py
│
├── .gitignore
└── README.md

Clases
0 -> Smoke
1 -> Fire

Instalación

Clonar repositorio:

git clone https://github.com/Jaume92/Proyecto_detector_humo_YoloV8.git
cd Proyecto_detector_humo_YoloV8


Instalar dependencias:

pip install ultralytics numpy==1.26.4

Entrenamiento

Ejecutar entrenamiento:

python training/train_model.py


El modelo entrenado se guarda en:

runs/detect/fire_smoke_v1/weights/best.pt

Inferencia (pendiente)

Se añadirá soporte para:

Webcam

RTSP (CCTV)

Vídeo local

Tecnologías

Python

YOLOv8 (Ultralytics)

PyTorch

OpenCV

Autor

Jaume Ruiz Marcos
GitHub: https://github.com/Jaume92