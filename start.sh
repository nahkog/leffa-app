#!/bin/bash

# API'yi başlat
uvicorn myapi:app --host 0.0.0.0 --port 8000 &

# Worker'ı başlat
python worker.py

# Gerekirse tail ile log açık tutmak istersen:
# tail -f /dev/null
