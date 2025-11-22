from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.backend.inference import predict_image

# app = FastAPI(root_path="/sketch")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    names, probas = predict_image(contents)
    return {"predictions": names, "probas": probas}

@app.get("/")
async def root():
    return {"message": "FastAPI backend is running."}
