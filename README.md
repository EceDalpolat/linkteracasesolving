# 🚗 Türkiye Otomotiv Satış Adet Tahmini API

Bu proje, Türkiye'deki otomotiv satış adetlerini tahmin etmek için hibrit makine öğrenmesi modeli (Time Series + Linear Regression) kullanır.

## 🎯 Canlı Demo
- **API URL**: `https://your-app.railway.app`
- **Health Check**: `https://your-app.railway.app/health`
- **Model Info**: `https://your-app.railway.app/model/info`

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
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "otv_orani": 15.5,
    "faiz": 17.0, 
    "eur_tl": 18.5,
    "kredi_stok": 85000000,
    "year": 2022,
    "month": 6
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