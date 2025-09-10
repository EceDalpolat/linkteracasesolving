"""
Otomotiv Satış Tahmini REST API
Flask ile RESTful API servisi
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import logging
from marshmallow import Schema, fields, ValidationError
import traceback

from src.models.automotive_regression import AutomotiveRegressionModel
from src.forecasting.time_series_forecaster import TimeSeriesForecaster

# Flask app setup
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None
ts_forecaster = None

class PredictionRequestSchema(Schema):
    """
    Tahmin isteği için validation schema
    """
    otv_orani = fields.Float(required=True, description="ÖTV Oranı (%)")
    faiz = fields.Float(required=True, description="Faiz Oranı (%)")
    eur_tl = fields.Float(required=True, description="EUR/TL Kuru")
    kredi_stok = fields.Float(required=True, description="Kredi Stok (Milyon TL)")
    date = fields.Date(required=True, description="Tahmin tarihi (YYYY-MM-DD)")

class BatchPredictionRequestSchema(Schema):
    """
    Toplu tahmin isteği için validation schema
    """
    predictions = fields.List(fields.Nested(PredictionRequestSchema), required=True)

def load_model():
    """
    Eğitilmiş modeli yükle
    """
    global model
    try:
        model = AutomotiveRegressionModel()
        model.load_model('models/automotive_regression_model.pkl')
        logger.info("Model başarıyla yüklendi ")
        return True
    except Exception as e:
        logger.error(f"Model yükleme hatası: {str(e)}")
        return False

def prepare_prediction_data(request_data):
    """
    API isteğinden gelen veriyi model için hazırla
    """
    try:
        # DataFrame oluştur
        df = pd.DataFrame([{
            'Date': pd.to_datetime(request_data['date']),
            'OTV Orani': request_data['otv_orani'],
            'Faiz': request_data['faiz'],
            'EUR/TL': request_data['eur_tl'],
            'Kredi Stok': request_data['kredi_stok'] * 1000000  # Milyon TL -> TL
        }])
        
        return df
    except Exception as e:
        logger.error(f"Veri hazırlama hatası: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """
    Servis sağlık kontrolü
    """
    return jsonify({
        'status': 'healthy',
        'service': 'Automotive Sales Prediction API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Model bilgilerini döndür
    """
    if model is None:
        return jsonify({'error': 'Model yüklenmedi'}), 500
    
    try:
        # Model özelliklerini al
        feature_importance = model.get_feature_importance().head(10).to_dict('records')
        
        return jsonify({
            'model_type': 'Multiple Linear Regression',
            'features_count': len(model.feature_names),
            'features': model.feature_names,
            'top_features': feature_importance,
            'is_fitted': model.is_fitted,
            'created_at': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Model bilgi hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Tek bir dönem için otomotiv satış tahmini
    """
    if model is None:
        return jsonify({'error': 'Model yüklenmedi'}), 500
    
    try:
        # Request validation
        schema = PredictionRequestSchema()
        try:
            request_data = schema.load(request.json)
        except ValidationError as err:
            return jsonify({'error': 'Geçersiz parametre', 'details': err.messages}), 400
        
        # Veriyi hazırla
        input_df = prepare_prediction_data(request_data)
        
        # Tahmin yap
        prediction_result = model.predict_future(input_df)
        predicted_value = float(prediction_result['Predicted_Otomotiv_Satis'].iloc[0])
        
        # Sonucu formatla
        response = {
            'prediction': {
                'date': request_data['date'].strftime('%Y-%m'),
                'predicted_sales': round(predicted_value, 0),
                'predicted_sales_formatted': f"{predicted_value:,.0f} adet"
            },
            'input_parameters': {
                'otv_orani': request_data['otv_orani'],
                'faiz': request_data['faiz'],
                'eur_tl': request_data['eur_tl'],
                'kredi_stok_million_tl': request_data['kredi_stok']
            },
            'model_info': {
                'type': 'Multiple Linear Regression',
                'features_used': len(model.feature_names)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Tahmin tamamlandı: {predicted_value:.0f} adet - {request_data['date']}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Tahmin hatası: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Tahmin hatası: {str(e)}'}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Çoklu dönem için otomotiv satış tahmini
    """
    if model is None:
        return jsonify({'error': 'Model yüklenmedi'}), 500
    
    try:
        # Request validation
        schema = BatchPredictionRequestSchema()
        try:
            request_data = schema.load(request.json)
        except ValidationError as err:
            return jsonify({'error': 'Geçersiz parametre', 'details': err.messages}), 400
        
        predictions = []
        
        for pred_request in request_data['predictions']:
            # Veriyi hazırla
            input_df = prepare_prediction_data(pred_request)
            
            # Tahmin yap
            prediction_result = model.predict_future(input_df)
            predicted_value = float(prediction_result['Predicted_Otomotiv_Satis'].iloc[0])
            
            predictions.append({
                'date': pred_request['date'].strftime('%Y-%m'),
                'predicted_sales': round(predicted_value, 0),
                'predicted_sales_formatted': f"{predicted_value:,.0f} adet",
                'input_parameters': {
                    'otv_orani': pred_request['otv_orani'],
                    'faiz': pred_request['faiz'],
                    'eur_tl': pred_request['eur_tl'],
                    'kredi_stok_million_tl': pred_request['kredi_stok']
                }
            })
        
        response = {
            'predictions': predictions,
            'summary': {
                'total_periods': len(predictions),
                'total_predicted_sales': sum([p['predicted_sales'] for p in predictions]),
                'average_monthly_sales': round(np.mean([p['predicted_sales'] for p in predictions]), 0)
            },
            'model_info': {
                'type': 'Multiple Linear Regression',
                'features_used': len(model.feature_names)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Toplu tahmin tamamlandı: {len(predictions)} dönem")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Toplu tahmin hatası: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Toplu tahmin hatası: {str(e)}'}), 500

@app.route('/predict/range', methods=['POST'])
def predict_date_range():
    """
    Belirli bir tarih aralığı için tahmin (sabit parametrelerle)
    """
    if model is None:
        return jsonify({'error': 'Model yüklenmedi'}), 500
    
    try:
        data = request.json
        
        # Gerekli parametreleri kontrol et
        required_params = ['start_date', 'end_date', 'otv_orani', 'faiz', 'eur_tl', 'kredi_stok']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f'Eksik parametre: {param}'}), 400
        
        # Tarih aralığını oluştur
        start_date = pd.to_datetime(data['start_date'])
        end_date = pd.to_datetime(data['end_date'])
        
        if start_date >= end_date:
            return jsonify({'error': 'Başlangıç tarihi bitiş tarihinden küçük olmalı'}), 400
        
        # Aylık tarih aralığı oluştur
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        if len(date_range) > 24:  # Maksimum 2 yıl
            return jsonify({'error': 'Maksimum 24 aylık tahmin yapılabilir'}), 400
        
        # Her ay için tahmin yap
        predictions = []
        
        for date in date_range:
            input_df = pd.DataFrame([{
                'Date': date,
                'OTV Orani': data['otv_orani'],
                'Faiz': data['faiz'],
                'EUR/TL': data['eur_tl'],
                'Kredi Stok': data['kredi_stok'] * 1000000
            }])
            
            prediction_result = model.predict_future(input_df)
            predicted_value = float(prediction_result['Predicted_Otomotiv_Satis'].iloc[0])
            
            predictions.append({
                'date': date.strftime('%Y-%m'),
                'predicted_sales': round(predicted_value, 0),
                'predicted_sales_formatted': f"{predicted_value:,.0f} adet"
            })
        
        response = {
            'date_range': {
                'start_date': start_date.strftime('%Y-%m'),
                'end_date': end_date.strftime('%Y-%m'),
                'total_months': len(predictions)
            },
            'parameters': {
                'otv_orani': data['otv_orani'],
                'faiz': data['faiz'],
                'eur_tl': data['eur_tl'],
                'kredi_stok_million_tl': data['kredi_stok']
            },
            'predictions': predictions,
            'summary': {
                'total_predicted_sales': sum([p['predicted_sales'] for p in predictions]),
                'average_monthly_sales': round(np.mean([p['predicted_sales'] for p in predictions]), 0),
                'min_monthly_sales': min([p['predicted_sales'] for p in predictions]),
                'max_monthly_sales': max([p['predicted_sales'] for p in predictions])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Tarih aralığı tahmini tamamlandı: {len(predictions)} ay")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Tarih aralığı tahmin hatası: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Tarih aralığı tahmin hatası: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'API endpoint bulunamadı'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Sunucu hatası'}), 500

if __name__ == '__main__':
    # Model yükle
    logger.info("Otomotiv Satış Tahmini API başlatılıyor...")
    
    if load_model():
        logger.info("API servisi başlatılıyor...")
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        logger.error("Model yüklenemedi, API başlatılamıyor")
        exit(1)
