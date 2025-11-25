from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.backend.inference import predict_image
from pathlib import Path

app = FastAPI(root_path="/sketch")
# app = FastAPI()

# Serve static files (CSS, JS, images)
static_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Serve static files (JS, CSS)
# app.mount("/static", StaticFiles(directory="src/app/frontend"), name="static")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    names, probas = predict_image(contents)
    return {"predictions": names, "probas": probas}

@app.get("/")
async def root():
    return {"message": "FastAPI backend is running."}

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    return FileResponse(str(static_path / "index.html"))
