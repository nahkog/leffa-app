FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential libgl1-mesa-glx git && \
    apt-get clean

# huggingface_hub sadece örnek script çalıştıracaksan kalabilir
RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads output

RUN chmod +x start.sh

CMD ["./start.sh"]
