import io
from contextlib import asynccontextmanager

import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from preprocess import CLASS_NAMES, preprocess

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["fashion"] = tf.keras.models.load_model("model.keras")  # dimuat sekali saat server start
    yield
    models.clear()


app = FastAPI(title="Fashion-MNIST Classifier API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid")

    x = preprocess(img)
    if x is None:
        raise HTTPException(status_code=422, detail="Tidak ada pakaian terdeteksi (gambar kosong)")

    probs = models["fashion"].predict(x, verbose=0)[0]
    return {
        "prediction": CLASS_NAMES[int(probs.argmax())],
        "confidence": float(probs.max()),
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }
