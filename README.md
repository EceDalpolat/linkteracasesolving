# 🚗 Türkiye Otomotiv Satış Adet Tahmini API

Bu proje, Türkiye'deki otomotiv satış adetlerini tahmin etmek için hibrit makine öğrenmesi modeli (Time Series + Linear Regression) kullanır.

## 🎯 Canlı Demo
- **API URL**: `https://linkteracasesolving-production.up.railway.app`
- **Health Check**: `https://linkteracasesolving-production.up.railway.app/health`
- **Model Info**: `https://linkteracasesolving-production.up.railway.app/model/info`

## 🚀 Hızlı Başlangıç

### 1. Kurulum
```bash
# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. Veriyi Yerleştir
`data/` klasörüne `Veri-Seti.xlsx` dosyasını koy.

### 3. Analiz Çalıştır
```bash
# Notebook ile
jupyter notebook automotive_sales_prediction.ipynb


### 4. API Başlat
```bash
python automotive_api.py
```

## 🔧 API Kullanım

**Base URL:** `http://localhost:8080`

### Endpoints:
- `GET /health` - Sistem durumu
- `GET /model/info` - Model bilgileri  
- `POST /predict` - Tek dönem tahmini
- `POST /predict/range` - Aralık tahmini

### Örnek Request:
```bash
curl -X POST https://linkteracasesolving-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2022-06-01",
    "values": {
      "OTV Orani": 65.0,
      "Faiz": 24.0,
      "EUR/TL": 17.5,
      "Kredi Stok": 5000000
    }
  }'
```

## Docker

```bash
# Build ve çalıştır
docker build -t automotive-api .
docker run -p 8080:8080 automotive-api

# Veya Docker Compose
docker-compose up
```

##Test

Postman'da `postman_collection.json` dosyasını import et ve test et.

##Dosya Yapısı

```
├── automotive_api.py              # API server
├── automotive_sales_prediction.ipynb  # Jupyter notebook  
├── src/                          # Kaynak kodlar
├── data/                         # Veri dosyaları
├── requirements.txt              # Bağımlılıklar
├── Dockerfile                    # Docker config
├── docker-compose.yml           # Docker Compose
└── postman_collection.json      # Postman testleri
```