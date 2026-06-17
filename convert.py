from ultralytics import YOLO

# 1. Cargar tu modelo YOLO
modelo = YOLO("best.pt")

# 2. Exportarlo al formato requerido por Vercel
modelo.export(format="onnx")