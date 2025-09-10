"""
Otomotiv Satış Tahmini için Çoklu Doğrusal Regresyon Modeli
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import get_logger
logger = get_logger()

class AutomotiveRegressionModel:
    """
    Otomotiv satış verisi için çoklu doğrusal regresyon modeli
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.stats_model = None
        self.model_summary = None
        
        logger.info("AutomotiveRegressionModel başlatıldı")
    
    def create_features(self, data):
        """
        Feature engineering - yeni özellikler oluştur
        """
        df = data.copy()
        
        # Tarih sütununu datetime'a çevir
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Zaman bazlı özellikler
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            
            # Mevsimsel değişkenler
            df['Is_Winter'] = df['Month'].isin([12, 1, 2]).astype(int)
            df['Is_Spring'] = df['Month'].isin([3, 4, 5]).astype(int)
            df['Is_Summer'] = df['Month'].isin([6, 7, 8]).astype(int)
            df['Is_Autumn'] = df['Month'].isin([9, 10, 11]).astype(int)
            
            # Yılsonu efekti
            df['Is_YearEnd'] = df['Month'].isin([11, 12]).astype(int)
            
            # Trend değişkeni
            df['Trend'] = range(len(df))
        
        # Ekonomik değişkenler arası etkileşimler
        if all(col in df.columns for col in ['Faiz', 'EUR/TL']):
            df['Faiz_EURTL_Interaction'] = df['Faiz'] * df['EUR/TL']
        
        if all(col in df.columns for col in ['OTV Orani', 'EUR/TL']):
            df['OTV_EURTL_Interaction'] = df['OTV Orani'] * df['EUR/TL']
        
        # Logaritmik dönüşümler (pozitif değerler için)
        numeric_cols = ['OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok']
        for col in numeric_cols:
            if col in df.columns and (df[col] > 0).all():
                df[f'{col}_Log'] = np.log(df[col])
        
        logger.info("Feature engineering tamamlandı")
        return df
    
    def prepare_data(self, data, target_col='Otomotiv Satis', 
                    feature_cols=None, add_features=True):
        """
        Veriyi model için hazırla
        """
        df = data.copy()
        
        if add_features:
            df = self.create_features(df)
        
        # Hedef değişken
        if target_col not in df.columns:
            raise ValueError(f"Hedef değişken '{target_col}' bulunamadı")
        
        y = df[target_col].copy()
        
        # Özellik sütunları
        if feature_cols is None:
            # Varsayılan özellikler
            feature_cols = [
                'OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok',
                'Year', 'Month', 'Quarter', 'Trend',
                'Is_Winter', 'Is_Spring', 'Is_Summer', 'Is_Autumn',
                'Is_YearEnd'
            ]
            
            # Etkileşim değişkenleri ekle
            interaction_cols = [col for col in df.columns if 'Interaction' in col]
            feature_cols.extend(interaction_cols)
            
            # Log değişkenleri ekle
            log_cols = [col for col in df.columns if '_Log' in col]
            feature_cols.extend(log_cols)
        
        # Mevcut sütunları filtrele
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].copy()
        
        # Eksik değerleri kaldır
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_indices]
        y = y[valid_indices]
        
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Veri hazırlandı: {len(X)} gözlem, {len(self.feature_names)} özellik")
        logger.info(f"Kullanılan özellikler: {self.feature_names}")
        
        return X, y
    
    def fit(self, X, y, scale_features=True):
        """
        Modeli eğit
        """
        if scale_features:
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X.copy()
        
        # Scikit-learn modeli
        self.model.fit(X_scaled, y)
        
        # Statsmodels modeli (detaylı analiz için)
        X_sm = sm.add_constant(X_scaled)
        self.stats_model = sm.OLS(y, X_sm).fit()
        self.model_summary = self.stats_model.summary()
        
        self.is_fitted = True
        
        logger.info("Model eğitimi tamamlandı ✅")
        return self
    
    def predict(self, X):
        """
        Tahmin yap
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def evaluate(self, X, y):
        """
        Model performansını değerlendir
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi")
        
        predictions = self.predict(X)
        
        metrics = {
            'R²': r2_score(y, predictions),
            'MAE': mean_absolute_error(y, predictions),
            'RMSE': np.sqrt(mean_squared_error(y, predictions)),
            'MAPE': np.mean(np.abs((y - predictions) / y)) * 100
        }
        
        logger.info("Model değerlendirmesi tamamlandı")
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Cross validation ile model performansını değerlendir
        """
        X_scaled = self.scaler.fit_transform(X)
        
        cv_scores = cross_val_score(self.model, X_scaled, y, 
                                   cv=cv, scoring='r2')
        
        cv_results = {
            'CV_R²_mean': cv_scores.mean(),
            'CV_R²_std': cv_scores.std(),
            'CV_scores': cv_scores
        }
        
        logger.info(f"Cross validation tamamlandı (CV={cv})")
        return cv_results
    
    def get_feature_importance(self):
        """
        Özellik önem derecelerini al
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_,
            'Abs_Coefficient': np.abs(self.model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        return importance_df
    
    def plot_diagnostics(self):
        """
        Model tanı grafikleri çiz
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Residuals vs Fitted
        fitted = self.stats_model.fittedvalues
        residuals = self.stats_model.resid
        
        axes[0, 0].scatter(fitted, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        
        # Q-Q Plot
        sm.qqplot(residuals, line='s', ax=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        
        # Scale-Location
        axes[1, 0].scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.6)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Residuals|')
        axes[1, 0].set_title('Scale-Location')
        
        # Feature Importance
        importance = self.get_feature_importance().head(10)
        axes[1, 1].barh(importance['Feature'], importance['Abs_Coefficient'])
        axes[1, 1].set_xlabel('|Coefficient|')
        axes[1, 1].set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        
        # Kaydet
        os.makedirs('data/processed/model_diagnostics', exist_ok=True)
        plt.savefig('data/processed/model_diagnostics/regression_diagnostics.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Model tanı grafikleri oluşturuldu")
    
    def statistical_tests(self):
        """
        İstatistiksel testler (heteroskedasticity, autocorrelation)
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi")
        
        results = {}
        
        # Durbin-Watson test (autocorrelation)
        dw_stat = durbin_watson(self.stats_model.resid)
        results['Durbin_Watson'] = dw_stat
        
        # Breusch-Pagan test (heteroskedasticity)
        bp_test = het_breuschpagan(self.stats_model.resid, self.stats_model.model.exog)
        results['Breusch_Pagan'] = {
            'LM_statistic': bp_test[0],
            'p_value': bp_test[1]
        }
        
        # White test (heteroskedasticity)
        white_test = het_white(self.stats_model.resid, self.stats_model.model.exog)
        results['White_Test'] = {
            'LM_statistic': white_test[0],
            'p_value': white_test[1]
        }
        
        logger.info("İstatistiksel testler tamamlandı")
        return results
    
    def save_model(self, filepath='models/automotive_regression_model.pkl'):
        """
        Modeli kaydet
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'stats_model': self.stats_model
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model kaydedildi: {filepath}")
        
        return filepath
    
    def load_model(self, filepath):
        """
        Modeli yükle
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.stats_model = model_data['stats_model']
        self.is_fitted = True
        
        logger.info(f"Model yüklendi: {filepath}")
        return self
    
    def predict_future(self, future_data):
        """
        Gelecek dönemler için tahmin yap
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi")
        
        # Feature engineering uygula
        future_df = self.create_features(future_data)
        
        # Sadece modelde kullanılan özellikleri al
        X_future = future_df[self.feature_names]
        
        # Tahmin yap
        predictions = self.predict(X_future)
        
        # Sonuçları DataFrame olarak döndür
        result_df = future_data.copy()
        result_df['Predicted_Otomotiv_Satis'] = predictions
        
        logger.info(f"Gelecek dönem tahminleri yapıldı: {len(predictions)} dönem")
        return result_df
