import asyncio
import json
import os
import uuid
import logging
from redis_conn import redis_client
from predictor import LeffaPredictor
from cloudflare_uploader import upload_to_r2
from PIL import Image
import model_initializer  # Yol ayarlarını ve gerekli importları yükler


# Çalışma klasörleri
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

predictor = LeffaPredictor()

async def process_job(job_data: dict):
    task_id = job_data["task_id"]
    job_type = job_data["type"]

    try:
        logger.info(f"🎯 Görev başlatıldı: {task_id} Tür: {job_type}")

        if job_type == "virtual_tryon":
            gen_image, _, _ = predictor.leffa_predict_vt(
                job_data["src_path"],
                job_data["tgt_path"],
                job_data["ref_acceleration"],
                job_data["step"],
                job_data["scale"],
                job_data["seed"],
                job_data["vt_model_type"],
                job_data["vt_garment_type"],
                job_data["vt_repaint"],
                job_data["preprocess_garment"]
            )

        elif job_type == "pose_transfer":
            gen_image, _, _ = predictor.leffa_predict_pt(
                job_data["src_path"],
                job_data["tgt_path"],
                job_data["ref_acceleration"],
                job_data["step"],
                job_data["scale"],
                job_data["seed"]
            )

        else:
            raise ValueError(f"Bilinmeyen görev türü: {job_type}")

        # Görseli geçici olarak kaydet
        output_filename = f"output_{task_id}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        Image.fromarray(gen_image).save(output_path)

        # Cloudflare R2'ye yükle ve link al
        r2_url = await upload_to_r2(output_path, output_filename)

        # Sonucu Redis'e yaz
        await redis_client.set(f"result:{task_id}", r2_url)
        logger.info(f"✅ Görev tamamlandı: {task_id} - URL: {r2_url}")

        # Dosyayı sil
        os.remove(output_path)

    except Exception as e:
        error_message = f"❌ Görev başarısız: {task_id} - {str(e)}"
        logger.error(error_message)
        await redis_client.set(f"result:{task_id}", f"error: {str(e)}")

async def worker_loop():
    logger.info("🚀 Worker başlatıldı ve kuyruğu dinliyor...")
    while True:
        job_raw = await redis_client.lpop("leffa_queue")
        if job_raw:
            job = json.loads(job_raw)
            await process_job(job)
        else:
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(worker_loop())
