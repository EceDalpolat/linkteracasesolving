"""
Ana Analiz ve Model Eğitimi
Türkiye Otomotiv Satış Adet Tahmini için Time Series + Multiple Linear Regression
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.forecasting.time_series_forecaster import TimeSeriesForecaster
from src.models.automotive_regression import AutomotiveRegressionModel
from src.utils.logger import get_logger

# Matplotlib Türkçe karakter desteği
plt.rcParams['font.family'] = ['DejaVu Sans']

logger = get_logger()

class AutomotiveAnalysis:
    """
    Türkiye Otomotiv Satış Analizi ve Tahmini
    """
    
    def __init__(self, data_path='data/raw/Veri-Seti.xlsx'):
        """
        Analiz sınıfını başlat
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.ts_forecaster = None
        self.regression_model = None
        self.forecast_results = None
        
        logger.info("AutomotiveAnalysis başlatıldı")
    
    def load_and_explore_data(self):
        """
        Veriyi yükle ve keşfedici analiz yap
        """
        logger.info("Veri yükleniyor ve analiz ediliyor...")
        
        # Veriyi yükle
        self.raw_data = pd.read_excel(self.data_path)
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
        
        print("=== VERİ SETİ BİLGİLERİ ===")
        print(f"Veri şekli: {self.raw_data.shape}")
        print(f"Tarih aralığı: {self.raw_data['Date'].min()} - {self.raw_data['Date'].max()}")
        print(f"Toplam ay sayısı: {len(self.raw_data)}")
        
        print("\n=== SÜTUN BİLGİLERİ ===")
        print(self.raw_data.dtypes)
        
        print("\n=== EKSİK DEĞER ANALİZİ ===")
        missing_info = self.raw_data.isnull().sum()
        missing_pct = (missing_info / len(self.raw_data)) * 100
        missing_df = pd.DataFrame({
            'Eksik_Sayi': missing_info,
            'Eksik_Yuzde': missing_pct
        })
        print(missing_df)
        
        print("\n=== İSTATİSTİKSEL ÖZETLİK ===")
        print(self.raw_data.describe())
        
        # Eksik değerlerin hangi dönemlerde olduğunu göster
        print("\n=== EKSİK VERİLERİN OLDUĞU DÖNEMLER ===")
        missing_periods = self.raw_data[self.raw_data.isnull().any(axis=1)]['Date']
        if not missing_periods.empty:
            print(f"İlk eksik dönem: {missing_periods.min()}")
            print(f"Son eksik dönem: {missing_periods.max()}")
            print(f"Eksik dönem sayısı: {len(missing_periods)}")
        
        return self.raw_data
    
    def visualize_data(self):
        """
        Veriyi görselleştir
        """
        logger.info("Veri görselleştiriliyor...")
        
        # Ana değişkenlerin zaman serisi grafikleri
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        variables = ['Otomotiv Satis', 'OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok']
        
        for i, var in enumerate(variables):
            row = i // 2
            col = i % 2
            
            # Eksik olmayan verileri çiz
            data_clean = self.raw_data.dropna(subset=[var])
            axes[row, col].plot(data_clean['Date'], data_clean[var], 
                              linewidth=2, color='blue', alpha=0.8)
            axes[row, col].set_title(f'{var} - Zaman Serisi', size=14, fontweight='bold')
            axes[row, col].set_xlabel('Tarih')
            axes[row, col].set_ylabel(var)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Eksik veri bölgelerini vurgula
            missing_data = self.raw_data[self.raw_data[var].isnull()]
            if not missing_data.empty:
                for date in missing_data['Date']:
                    axes[row, col].axvline(x=date, color='red', alpha=0.3, linestyle='--')
        
        # Son subplot'u kaldır
        axes[2, 1].remove()
        
        plt.tight_layout()
        
        # Kaydet
        os.makedirs('data/processed/analysis_plots', exist_ok=True)
        plt.savefig('data/processed/analysis_plots/time_series_overview.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Korelasyon matrisi
        plt.figure(figsize=(12, 8))
        corr_data = self.raw_data[variables].corr()
        sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Değişkenler Arası Korelasyon Matrisi', size=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/processed/analysis_plots/correlation_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Veri görselleştirme tamamlandı")
    
    def forecast_independent_variables(self, periods=13):
        """
        Bağımsız değişkenler için time series tahmini yap
        """
        logger.info("Bağımsız değişkenler için time series tahmini başlatılıyor...")
        
        # Sadece eksik olmayan verileri kullan (2022-06'ya kadar)
        complete_data = self.raw_data.dropna()
        
        # TimeSeriesForecaster'ı başlat
        self.ts_forecaster = TimeSeriesForecaster(complete_data, date_col='Date')
        
        # Bağımsız değişkenleri tahmin et
        independent_vars = ['OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok']
        forecast_results = self.ts_forecaster.forecast_all_variables(
            independent_vars, periods=periods, seasonal=True
        )
        
        # Sonuçları göster
        print("=== BAĞIMSIZ DEĞİŞKEN TAHMİNLERİ ===")
        for var, forecast in forecast_results.items():
            print(f"\n{var} tahmini:")
            print(forecast.round(4))
            
            # Grafik çiz
            self.ts_forecaster.plot_forecast(var, last_n_points=36)
        
        # Tahminleri kaydet
        forecast_df = self.ts_forecaster.get_forecast_dataframe()
        if forecast_df is not None:
            self.ts_forecaster.save_forecasts()
            
            # Tahmin edilen verileri ana veri setine ekle
            self.processed_data = self.raw_data.copy()
            
            # Eksik dönemleri tahmin edilen verilerle doldur
            for var in independent_vars:
                if var in forecast_df.columns:
                    missing_mask = self.processed_data[var].isnull()
                    missing_dates = self.processed_data.loc[missing_mask, 'Date']
                    
                    for date in missing_dates:
                        if date in forecast_df.index:
                            self.processed_data.loc[
                                self.processed_data['Date'] == date, var
                            ] = forecast_df.loc[date, var]
            
            logger.info("Bağımsız değişken tahminleri tamamlandı ✅")
            return forecast_df
        
        return None
    
    def train_regression_model(self):
        """
        Çoklu doğrusal regresyon modelini eğit
        """
        logger.info("Çoklu doğrusal regresyon modeli eğitiliyor...")
        
        if self.processed_data is None:
            raise ValueError("Önce bağımsız değişken tahminleri yapılmalı")
        
        # Sadece hedef değişkeni eksik olmayan veriler
        training_data = self.processed_data[
            self.processed_data['Otomotiv Satis'].notna()
        ].copy()
        
        print(f"Eğitim veri sayısı: {len(training_data)}")
        
        # Regresyon modelini başlat
        self.regression_model = AutomotiveRegressionModel()
        
        # Veriyi hazırla
        X, y = self.regression_model.prepare_data(
            training_data, 
            target_col='Otomotiv Satis',
            add_features=True
        )
        
        print(f"Model eğitimi: {len(X)} gözlem, {len(X.columns)} özellik")
        
        # Modeli eğit
        self.regression_model.fit(X, y, scale_features=True)
        
        # Model performansını değerlendir
        metrics = self.regression_model.evaluate(X, y)
        
        print("\n=== MODEL PERFORMANSI ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Cross validation
        cv_results = self.regression_model.cross_validate(X, y, cv=5)
        print(f"\nCross Validation R²: {cv_results['CV_R²_mean']:.4f} ± {cv_results['CV_R²_std']:.4f}")
        
        # Özellik önemliliği
        print("\n=== ÖZELLİK ÖNEMLİLİĞİ (Top 10) ===")
        importance = self.regression_model.get_feature_importance().head(10)
        print(importance[['Feature', 'Coefficient']].to_string(index=False))
        
        # Model istatistikleri
        print("\n=== DETAYLI MODEL İSTATİSTİKLERİ ===")
        print(self.regression_model.model_summary)
        
        # İstatistiksel testler
        stat_tests = self.regression_model.statistical_tests()
        print("\n=== İSTATİSTİKSEL TESTLER ===")
        print(f"Durbin-Watson: {stat_tests['Durbin_Watson']:.4f}")
        print(f"Breusch-Pagan p-value: {stat_tests['Breusch_Pagan']['p_value']:.4f}")
        print(f"White Test p-value: {stat_tests['White_Test']['p_value']:.4f}")
        
        # Tanı grafikleri
        self.regression_model.plot_diagnostics()
        
        # Modeli kaydet
        self.regression_model.save_model()
        
        logger.info("Regresyon modeli eğitimi tamamlandı ")
        
        return metrics
    
    def predict_target_variable(self):
        """
        Hedef değişken (Otomotiv Satış) tahminini yap
        """
        logger.info("Hedef değişken tahmini yapılıyor...")
        
        if self.regression_model is None:
            raise ValueError("Önce regresyon modeli eğitilmeli")
        
        # Tahmin edilecek dönemler (eksik hedef değişkeni olan)
        prediction_data = self.processed_data[
            self.processed_data['Otomotiv Satis'].isna()
        ].copy()
        
        if len(prediction_data) == 0:
            logger.warning("Tahmin edilecek dönem bulunamadı")
            return None
        
        print(f"Tahmin edilecek dönem sayısı: {len(prediction_data)}")
        
        # Tahmin yap
        result_df = self.regression_model.predict_future(prediction_data)
        
        print("\n=== OTOMOTIV SATIŞ TAHMİNLERİ ===")
        result_display = result_df[['Date', 'Predicted_Otomotiv_Satis']].copy()
        result_display['Date'] = result_display['Date'].dt.strftime('%Y-%m')
        print(result_display.to_string(index=False))
        
        # Sonuçları kaydet
        os.makedirs('data/processed/predictions', exist_ok=True)
        result_df.to_csv('data/processed/predictions/automotive_sales_predictions.csv', index=False)
        
        # Tahmin grafiği çiz
        self.plot_final_predictions(result_df)
        
        self.forecast_results = result_df
        
        logger.info("Hedef değişken tahmini tamamlandı ")
        
        return result_df
    
    def plot_final_predictions(self, predictions_df):
        """
        Final tahmin sonuçlarını görselleştir
        """
        plt.figure(figsize=(20, 10))
        
        # Geçmiş veriler
        historical_data = self.raw_data[self.raw_data['Otomotiv Satis'].notna()]
        plt.plot(historical_data['Date'], historical_data['Otomotiv Satis'], 
                label='Gerçek Veriler', linewidth=3, color='blue', alpha=0.8)
        
        # Tahminler
        plt.plot(predictions_df['Date'], predictions_df['Predicted_Otomotiv_Satis'], 
                label='Tahmin Edilen Değerler', linewidth=3, color='red', 
                linestyle='--', marker='o', markersize=8)
        
        # Son gerçek veri noktasını vurgula
        last_real = historical_data.iloc[-1]
        last_date_str = last_real['Date'].strftime("%Y-%m")
        plt.scatter(last_real['Date'], last_real['Otomotiv Satis'], 
                   color='green', s=200, zorder=5, 
                   label=f'Son Gerçek Veri ({last_date_str})')
        
        plt.title('Türkiye Otomotiv Satış Adet Tahmini (Haz\'22 - Haz\'23)', 
                 size=18, fontweight='bold', pad=20)
        plt.xlabel('Tarih', size=14)
        plt.ylabel('Otomotiv Satış Adedi', size=14)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Y ekseni formatı
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        plt.tight_layout()
        
        # Kaydet
        plt.savefig('data/processed/predictions/automotive_sales_forecast.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Final tahmin grafiği oluşturuldu")
    
    def generate_summary_report(self):
        """
        Özet rapor oluştur
        """
        logger.info("Özet rapor oluşturuluyor...")
        
        report = f"""
# TÜRKİYE OTOMOTİV SATIŞ TAHMİNİ RAPORU
Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## VERİ SETİ BİLGİLERİ
- Toplam Gözlem: {len(self.raw_data)}
- Tarih Aralığı: {self.raw_data['Date'].min().strftime('%Y-%m')} - {self.raw_data['Date'].max().strftime('%Y-%m')}
- Eksik Dönem: {self.raw_data['Otomotiv Satis'].isnull().sum()} ay

## KULLANILAN YÖNTEM
1. **Time Series Forecasting**: Bağımsız değişkenler (ÖTV, Faiz, EUR/TL, Kredi Stok) için ARIMA/SARIMA modelleri
2. **Multiple Linear Regression**: Hedef değişken (Otomotiv Satış) tahmini için çoklu doğrusal regresyon

## MODEL PERFORMANSI
"""
        
        if self.regression_model and self.regression_model.is_fitted:
            # Eğitim verisinden metrikleri al
            training_data = self.processed_data[self.processed_data['Otomotiv Satis'].notna()].copy()
            X, y = self.regression_model.prepare_data(training_data, target_col='Otomotiv Satis')
            metrics = self.regression_model.evaluate(X, y)
            
            report += f"""
- R² Score: {metrics['R²']:.4f}
- Mean Absolute Error (MAE): {metrics['MAE']:,.0f}
- Root Mean Square Error (RMSE): {metrics['RMSE']:,.0f}
- Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%
"""
        
        if self.forecast_results is not None:
            report += f"""
## TAHMİN SONUÇLARI (Haz'22 - Haz'23)
"""
            for _, row in self.forecast_results.iterrows():
                report += f"- {row['Date'].strftime('%Y-%m')}: {row['Predicted_Otomotiv_Satis']:,.0f} adet\n"
        
        report += f"""
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
"""
        
        # Raporu kaydet
        os.makedirs('data/processed/reports', exist_ok=True)
        with open('data/processed/reports/analysis_summary.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        logger.info("Özet rapor oluşturuldu: data/processed/reports/analysis_summary.md")
        
        return report
    
    def run_complete_analysis(self):
        """
        Tüm analizi çalıştır
        """
        logger.info("Tam analiz başlatılıyor...")
        
        try:
            # 1. Veri yükleme ve keşif
            self.load_and_explore_data()
            
            # 2. Veri görselleştirme
            self.visualize_data()
            
            # 3. Bağımsız değişken tahminleri
            self.forecast_independent_variables(periods=13)
            
            # 4. Regresyon modeli eğitimi
            self.train_regression_model()
            
            # 5. Hedef değişken tahmini
            self.predict_target_variable()
            
            # 6. Özet rapor
            self.generate_summary_report()
            
            logger.info("Tam analiz başarıyla tamamlandı ")
            
        except Exception as e:
            logger.error(f"Analiz sırasında hata: {str(e)}")
            raise

if __name__ == "__main__":
    # Ana analizi çalıştır
    analysis = AutomotiveAnalysis()
    analysis.run_complete_analysis()
