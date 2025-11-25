from fastapi import FastAPI, File, UploadFile
from app.backend.inference import predict_image

app = FastAPI(root_path="/sketch")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    names, probas = predict_image(contents)
    return {"predictions": names, "probas": probas}

@app.get("/")
async def root():
    return {"message": "FastAPI backend is running."}

