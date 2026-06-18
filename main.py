from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
import cv2

app = FastAPI()

# Permitir peticiones desde tu frontend
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

session = ort.InferenceSession("best.onnx")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
    input_tensor = np.expand_dims(np.transpose(img / 255.0, (2, 0, 1)).astype(np.float32), 0)
    
    result = session.run(None, {session.get_inputs()[0].name: input_tensor})[0]
    label = "Podrida" if np.argmax(result) == 0 else "Fresca"
    return {"resultado": label}

@app.get("/predict")
def read_predict():
    return {"mensaje": "La API está lista. Envía una petición POST con una imagen a este mismo endpoint."}