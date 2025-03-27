#!/bin/bash

MODEL_DIR="/workspace/ckpts"
FLAG_FILE="$MODEL_DIR/.model_downloaded"

if [ ! -f "$FLAG_FILE" ]; then
  echo "📥 Model indiriliyor..."
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='franciszzj/Leffa', local_dir='$MODEL_DIR')"
  touch "$FLAG_FILE"
  echo "✅ Model indirildi ve kaydedildi."
else
  echo "🚀 Model zaten mevcut, indirilmeyecek."
fi

# API ve worker'ı başlat
bash /workspace/start.sh