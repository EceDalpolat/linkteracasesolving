from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Global değişkenler
model_data = None
regression_model = None
arima_models = None
feature_columns = None

def load_model():
    """Modeli yükle"""
    global model_data, regression_model, arima_models, feature_columns
    try:
        model_data = joblib.load('models/automotive_prediction_model.pkl')
        regression_model = model_data['regression_model']
        arima_models = model_data['arima_models']
        feature_columns = model_data['feature_columns']
        print("✅ Model başarıyla yüklendi")
        return True
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return False

def predict_independent_variables(date, current_values=None):
    """Bağımsız değişkenleri ARIMA ile tahmin et"""
    predictions = {}
    
    # Eğer değerler verilmişse onları kullan, yoksa ARIMA tahmin et
    for var in model_data['independent_vars']:
        if current_values and var in current_values:
            predictions[var] = current_values[var]
        elif var in arima_models:
            # ARIMA ile tahmin (basitleştirilmiş)
            model = arima_models[var]
            # Son bilinen değerden tahmin yap
            forecast = model.predict(n_periods=1)[0]
            predictions[var] = forecast
        else:
            # Varsayılan değer
            defaults = {
                'OTV Orani': 65.0,
                'Faiz': 24.0,
                'EUR/TL': 17.0,
                'Kredi Stok': 5000000
            }
            predictions[var] = defaults.get(var, 0)
    
    return predictions

@app.route('/health', methods=['GET'])
def health():
    """API sağlık kontrolü"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_data is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Model bilgilerini döndür"""
    if model_data is None:
        return jsonify({'error': 'Model yüklenmedi'}), 500
    
    performance = model_data['model_performance']
    return jsonify({
        'model_type': 'Time Series + Linear Regression',
        'features': feature_columns,
        'arima_models': {var: str(model.order) for var, model in arima_models.items()},
        'performance': {
            'r2_score': round(performance['r2'], 4),
            'mape': round(performance['mape'], 2)
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Tahmin yap"""
    if model_data is None:
        return jsonify({'error': 'Model yüklenmedi'}), 500
    
    try:
        data = request.json
        
        # Gerekli parametreler
        required_params = ['date']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f'Eksik parametre: {param}'}), 400
        
        # Tarih işle
        predict_date = pd.to_datetime(data['date'])
        
        # Bağımsız değişkenleri al veya tahmin et
        independent_predictions = predict_independent_variables(
            predict_date, 
            current_values=data.get('values', {})
        )
        
        # Özellik vektörü oluştur
        features = {}
        
        # Bağımsız değişkenler
        for var in model_data['independent_vars']:
            features[var] = independent_predictions[var]
        
        # Tarih özellikleri
        features['Month'] = predict_date.month
        features['Year'] = predict_date.year
        
        # Tahmin için DataFrame oluştur
        X_pred = pd.DataFrame([features])[feature_columns]
        
        # Tahmin yap
        prediction = regression_model.predict(X_pred)[0]
        
        return jsonify({
            'prediction': {
                'date': predict_date.strftime('%Y-%m'),
                'predicted_sales': round(prediction, 0),
                'predicted_sales_formatted': f"{prediction:,.0f} adet"
            },
            'inputs': {
                'independent_variables': independent_predictions,
                'features_used': features
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Tahmin hatası: {str(e)}'}), 500

@app.route('/predict/range', methods=['POST'])
def predict_range():
    """Tarih aralığı için tahmin"""
    if model_data is None:
        return jsonify({'error': 'Model yüklenmedi'}), 500
    
    try:
        data = request.json
        
        # Parametreler
        start_date = pd.to_datetime(data['start_date'])
        end_date = pd.to_datetime(data['end_date'])
        values = data.get('values', {})
        
        # Aylık tarih aralığı
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        predictions = []
        for date in date_range:
            # Her ay için tahmin
            independent_pred = predict_independent_variables(date, values)
            
            features = {}
            for var in model_data['independent_vars']:
                features[var] = independent_pred[var]
            features['Month'] = date.month
            features['Year'] = date.year
            
            X_pred = pd.DataFrame([features])[feature_columns]
            prediction = regression_model.predict(X_pred)[0]
            
            predictions.append({
                'date': date.strftime('%Y-%m'),
                'predicted_sales': round(prediction, 0)
            })
        
        return jsonify({
            'predictions': predictions,
            'summary': {
                'total_months': len(predictions),
                'total_sales': sum([p['predicted_sales'] for p in predictions]),
                'average_monthly': round(np.mean([p['predicted_sales'] for p in predictions]), 0)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Tahmin hatası: {str(e)}'}), 500

# Model'i application startup'ta yükle
print("🚀 Otomotiv Satış Tahmini API başlatılıyor...")
load_model()

if __name__ == '__main__':
    if model_data is not None:
        print("📡 API http://localhost:8080 adresinde çalışıyor")
        app.run(host='0.0.0.0', port=8080, debug=True)
    else:
        print("❌ Model yüklenemedi!")
