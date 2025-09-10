"""
Time Series Forecasting Module for Independent Variables
Bu modül bağımsız değişkenler için ARIMA/SARIMA modellerini uygular.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import get_logger
logger = get_logger()

class TimeSeriesForecaster:
    """
    Bağımsız değişkenler için time series forecasting sınıfı
    """
    
    def __init__(self, data, date_col='Date'):
        """
        Initialize forecaster
        
        Parameters:
        -----------
        data : pd.DataFrame
            Zaman serisi verisi
        date_col : str
            Tarih sütunu adı
        """
        self.data = data.copy()
        self.date_col = date_col
        self.models = {}
        self.forecasts = {}
        
        # Tarih sütununu datetime'a çevir ve index olarak ayarla
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        self.data.set_index(date_col, inplace=True)
        
        logger.info("TimeSeriesForecaster başlatıldı")
    
    def check_stationarity(self, series, series_name):
        """
        Durağanlık testleri (ADF ve KPSS)
        """
        logger.info(f"{series_name} için durağanlık testi yapılıyor...")
        
        # ADF Test (H0: Non-stationary)
        adf_result = adfuller(series.dropna())
        adf_pvalue = adf_result[1]
        
        # KPSS Test (H0: Stationary)
        kpss_result = kpss(series.dropna())
        kpss_pvalue = kpss_result[1]
        
        logger.info(f"{series_name} - ADF p-value: {adf_pvalue:.4f}")
        logger.info(f"{series_name} - KPSS p-value: {kpss_pvalue:.4f}")
        
        # Stationarity decision
        if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
            logger.info(f"{series_name} durağan ")
            return True
        else:
            logger.info(f"{series_name} durağan değil ")
            return False
    
    def fit_auto_arima(self, series, series_name, seasonal=True, m=12):
        """
        Auto ARIMA ile optimal parametreleri bul
        """
        logger.info(f"{series_name} için Auto ARIMA modeli aranıyor...")
        
        try:
            model = auto_arima(
                series.dropna(),
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                seasonal=seasonal,
                m=m,
                start_P=0, start_Q=0,
                max_P=3, max_Q=3,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )
            
            logger.info(f"{series_name} - Optimal ARIMA: {model.order}")
            if seasonal:
                logger.info(f"{series_name} - Seasonal parameters: {model.seasonal_order}")
            
            return model
            
        except Exception as e:
            logger.error(f"{series_name} Auto ARIMA hatası: {str(e)}")
            return None
    
    def forecast_variable(self, variable_name, periods=13, seasonal=True):
        """
        Tek bir değişken için tahmin yap
        """
        logger.info(f"{variable_name} için tahmin başlatılıyor...")
        
        # Veriyi hazırla
        series = self.data[variable_name].dropna()
        
        if len(series) < 24:  # En az 2 yıl veri gerekli
            logger.warning(f"{variable_name} için yeterli veri yok")
            return None
        
        # Durağanlık kontrolü
        is_stationary = self.check_stationarity(series, variable_name)
        
        # Model fit et
        model = self.fit_auto_arima(series, variable_name, seasonal=seasonal)
        
        if model is None:
            return None
        
        # Tahmin yap
        try:
            forecast = model.predict(n_periods=periods)
            forecast_index = pd.date_range(
                start=series.index[-1] + pd.DateOffset(months=1),
                periods=periods,
                freq='MS'
            )
            
            forecast_series = pd.Series(forecast, index=forecast_index)
            
            # Modeli ve tahmini kaydet
            self.models[variable_name] = model
            self.forecasts[variable_name] = forecast_series
            
            logger.info(f"{variable_name} tahmini tamamlandı ")
            
            return forecast_series
            
        except Exception as e:
            logger.error(f"{variable_name} tahmin hatası: {str(e)}")
            return None
    
    def forecast_all_variables(self, variables, periods=13, seasonal=True):
        """
        Tüm bağımsız değişkenler için tahmin yap
        """
        logger.info("Tüm bağımsız değişkenler için tahmin başlatılıyor...")
        
        results = {}
        
        for var in variables:
            if var in self.data.columns:
                forecast = self.forecast_variable(var, periods, seasonal)
                if forecast is not None:
                    results[var] = forecast
            else:
                logger.warning(f"{var} veri setinde bulunamadı")
        
        return results
    
    def plot_forecast(self, variable_name, last_n_points=36):
        """
        Tahmin sonuçlarını görselleştir
        """
        if variable_name not in self.forecasts:
            logger.error(f"{variable_name} için tahmin bulunamadı")
            return
        
        plt.figure(figsize=(15, 8))
        
        # Orijinal veri
        original = self.data[variable_name].dropna()[-last_n_points:]
        plt.plot(original.index, original.values, 
                label='Gerçek Veriler', linewidth=2, color='blue')
        
        # Tahmin
        forecast = self.forecasts[variable_name]
        plt.plot(forecast.index, forecast.values, 
                label='Tahmin', linewidth=2, color='red', linestyle='--')
        
        plt.title(f'{variable_name} - Time Series Forecast', size=16, fontweight='bold')
        plt.xlabel('Tarih', size=12)
        plt.ylabel(variable_name, size=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Grafik kaydet
        os.makedirs('data/processed/forecast_plots', exist_ok=True)
        # Dosya adını güvenli hale getir
        safe_name = variable_name.replace('/', '_').replace(' ', '_')
        plt.savefig(f'data/processed/forecast_plots/{safe_name}_forecast.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"{variable_name} tahmin grafiği kaydedildi")
    
    def get_forecast_dataframe(self):
        """
        Tüm tahminleri tek bir DataFrame'de birleştir
        """
        if not self.forecasts:
            logger.warning("Henüz tahmin yapılmadı")
            return None
        
        forecast_df = pd.DataFrame(self.forecasts)
        forecast_df.index.name = 'Date'
        
        logger.info("Tahmin DataFrame'i oluşturuldu")
        return forecast_df
    
    def save_forecasts(self, file_path='data/processed/independent_variables_forecast.csv'):
        """
        Tahminleri CSV dosyasına kaydet
        """
        forecast_df = self.get_forecast_dataframe()
        
        if forecast_df is not None:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            forecast_df.to_csv(file_path)
            logger.info(f"Tahminler kaydedildi: {file_path}")
            return file_path
        
        return None
    
    def evaluate_model(self, variable_name, test_periods=12):
        """
        Model performansını değerlendir (geçmişe dönük tahmin)
        """
        if variable_name not in self.data.columns:
            logger.error(f"{variable_name} bulunamadı")
            return None
        
        series = self.data[variable_name].dropna()
        
        if len(series) < test_periods + 24:
            logger.warning(f"{variable_name} için yeterli veri yok")
            return None
        
        # Train/test split
        train = series[:-test_periods]
        test = series[-test_periods:]
        
        # Model fit
        model = auto_arima(train, stepwise=True, suppress_warnings=True)
        
        # Tahmin
        forecast = model.predict(n_periods=test_periods)
        
        # Metrikler
        mae = np.mean(np.abs(test.values - forecast))
        mape = np.mean(np.abs((test.values - forecast) / test.values)) * 100
        rmse = np.sqrt(np.mean((test.values - forecast) ** 2))
        
        results = {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'Model': model.order
        }
        
        logger.info(f"{variable_name} model değerlendirmesi tamamlandı")
        return results
