from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os
import logging
import json
from redis_conn import redis_client
from PIL import Image
import io

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

MAX_FILE_SIZE_MB = 10  # Maksimum dosya boyutu (MB)

async def save_uploaded_file(upload_file: UploadFile) -> str:
    contents = await upload_file.read()

    # Dosya format kontrolü
    if not upload_file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Sadece PNG veya JPEG format desteklenir.")

    # Dosya boyutu kontrolü
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Dosya boyutu {MAX_FILE_SIZE_MB}MB'den büyük olamaz.")

    # Görseli yeniden boyutlandır (isteğe bağlı)
    try:
        image = Image.open(io.BytesIO(contents))
        image = image.resize((768, 1024))
        filename = f"{uuid.uuid4().hex}_{upload_file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        image.save(filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Görsel işleme hatası: {e}")

    logger.info(f"Dosya kaydedildi: {filepath}, Boyut: {len(contents)} byte")
    return filepath

@app.post("/virtual_tryon")
async def virtual_tryon(
    src_img: UploadFile = File(...),
    target_img: UploadFile = File(...),
    ref_acceleration: bool = Form(False),
    step: int = Form(50),
    scale: float = Form(2.5),
    seed: int = Form(42),
    vt_model_type: str = Form("viton_hd"),
    vt_garment_type: str = Form("upper_body"),
    vt_repaint: bool = Form(False),
    preprocess_garment: bool = Form(False)
):
    logger.info("\U0001F680 /virtual_tryon isteği alındı")
    src_path = await save_uploaded_file(src_img)
    tgt_path = await save_uploaded_file(target_img)

    task_id = str(uuid.uuid4())
    job = {
        "task_id": task_id,
        "type": "virtual_tryon",
        "src_path": src_path,
        "tgt_path": tgt_path,
        "ref_acceleration": ref_acceleration,
        "step": step,
        "scale": scale,
        "seed": seed,
        "vt_model_type": vt_model_type,
        "vt_garment_type": vt_garment_type,
        "vt_repaint": vt_repaint,
        "preprocess_garment": preprocess_garment
    }
    await redis_client.rpush("leffa_queue", json.dumps(job))
    return {"task_id": task_id, "status": "queued"}

@app.post("/pose_transfer")
async def pose_transfer(
    src_img: UploadFile = File(...),
    target_img: UploadFile = File(...),
    ref_acceleration: bool = Form(False),
    step: int = Form(50),
    scale: float = Form(2.5),
    seed: int = Form(42)
):
    logger.info("\U0001F680 /pose_transfer isteği alındı")
    src_path = await save_uploaded_file(src_img)
    tgt_path = await save_uploaded_file(target_img)

    task_id = str(uuid.uuid4())
    job = {
        "task_id": task_id,
        "type": "pose_transfer",
        "src_path": src_path,
        "tgt_path": tgt_path,
        "ref_acceleration": ref_acceleration,
        "step": step,
        "scale": scale,
        "seed": seed
    }
    await redis_client.rpush("leffa_queue", json.dumps(job))
    return {"task_id": task_id, "status": "queued"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    result = await redis_client.get(f"result:{task_id}")
    if result is None:
        return JSONResponse(content={"status": "processing", "message": "Henüz tamamlanmadı"}, status_code=202)
    return JSONResponse(content={"status": "done", "result": result})

if __name__ == "__main__":
    import uvicorn
    logger.info("\u2728 Sunucu başlatılıyor...")
    uvicorn.run("myapi:app", host="0.0.0.0", port=8000, reload=True)
