
# TÜRKİYE OTOMOTİV SATIŞ TAHMİNİ RAPORU
Tarih: 2025-09-08 16:00

## VERİ SETİ BİLGİLERİ
- Toplam Gözlem: 162
- Tarih Aralığı: 2010-01 - 2023-06
- Eksik Dönem: 13 ay

## KULLANILAN YÖNTEM
1. **Time Series Forecasting**: Bağımsız değişkenler (ÖTV, Faiz, EUR/TL, Kredi Stok) için ARIMA/SARIMA modelleri
2. **Multiple Linear Regression**: Hedef değişken (Otomotiv Satış) tahmini için çoklu doğrusal regresyon

## MODEL PERFORMANSI

- R² Score: 0.8078
- Mean Absolute Error (MAE): 8,621
- Root Mean Square Error (RMSE): 11,436
- Mean Absolute Percentage Error (MAPE): 15.34%

## TAHMİN SONUÇLARI (Haz'22 - Haz'23)
- 2022-06: 122,427 adet
- 2022-07: 118,424 adet
- 2022-08: 132,210 adet
- 2022-09: 132,529 adet
- 2022-10: 130,172 adet
- 2022-11: 151,509 adet
- 2022-12: 195,218 adet
- 2023-01: 111,627 adet
- 2023-02: 126,498 adet
- 2023-03: 147,924 adet
- 2023-04: 145,875 adet
- 2023-05: 160,794 adet
- 2023-06: 154,289 adet

## DOSYA KONUMLARI
- Model: models/automotive_regression_model.pkl
- Tahminler: data/processed/predictions/automotive_sales_predictions.csv
- Grafikler: data/processed/analysis_plots/ ve data/processed/predictions/
- Bağımsız değişken tahminleri: data/processed/independent_variables_forecast.csv

## ÖNERİLER
1. Model performansını artırmak için daha fazla dış veri kaynağı kullanılabilir
2. Sezonsal etkiler ve trend analizi derinleştirilebilir
3. Ensemble modeller ile tahmin doğruluğu artırılabilir
4. Model düzenli olarak güncellenmelidir
