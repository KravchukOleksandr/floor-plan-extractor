import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from .pipeline import FloorPlanExtractor
from .postprocess import PRESETS


@asynccontextmanager
async def lifespan(app):
    app.state.extractor = FloorPlanExtractor(
        os.getenv("MODEL_PATH"), os.getenv("DEVICE", "auto"), os.getenv("DEPTH_MODEL")
    )
    yield


app = FastAPI(title="Floor Plan Extractor", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(app.state.extractor.device)}


@app.get("/presets")
def presets():
    return PRESETS


@app.post("/extract")
async def extract(image: UploadFile = File(...), preset: str = Query("weak")):
    if preset not in PRESETS:
        raise HTTPException(400, f"Unknown preset: {preset}")
    try:
        source = Image.open(BytesIO(await image.read())).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "Unsupported image") from error
    result = app.state.extractor.extract(source, preset)
    name = f"{Path(image.filename or 'result').stem}_result.zip"
    return Response(result.archive(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
