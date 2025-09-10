# Otomotiv Satış Tahmini API - Docker Image
FROM python:3.9-slim

# Çalışma dizini oluştur
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Port 8080'i dışa aç
EXPOSE 8080

# Çevre değişkenleri
ENV FLASK_APP=automotive_api.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Gunicorn ile uygulamayı başlat (Railway PORT env variable kullan)
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --timeout 120 automotive_api:app
