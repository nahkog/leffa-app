import os
import logging

R2_ACCESS_KEY = "17fc8526cc0cc7c82c6dd42a6aa4daa1"
R2_SECRET_KEY = "3c8b4a5eaac37769ba8e3a64a2c0570c09d82be944ef92ab82495f158b78a8a0"
R2_ENDPOINT = "https://a6c4993edecebe502f0fa66a1504bbf3.r2.cloudflarestorage.com"
R2_BUCKET_NAME = "leffabucket"
R2_PUBLIC_URL = "https://pub-f8a5d7b190664d6090737bf1663b7340.r2.dev"

import boto3
from botocore.client import Config
import asyncio


# R2 istemcisi kur
session = boto3.session.Session()
s3_client = session.client(
    service_name="s3",
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    endpoint_url=R2_ENDPOINT,
    config=Config(signature_version="s3v4")
)

async def upload_to_r2(file_path: str, file_name: str) -> str:
    loop = asyncio.get_event_loop()

    def upload():
        with open(file_path, "rb") as f:
            s3_client.upload_fileobj(f, R2_BUCKET_NAME, file_name)

    await loop.run_in_executor(None, upload)

    public_url = f"{R2_PUBLIC_URL}/{file_name}"
    return public_url
