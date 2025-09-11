"""
Web Verification Interface for Flower Detection
Flask-based web application for testing and verifying model predictions.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, List
import requests

from ..inference.inference_engine import FlowerInferenceEngine

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'flower_detection_secret_key_2024'  # Change in production

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global inference engine
inference_engine: FlowerInferenceEngine = None

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_inference_engine():
    """Initialize the inference engine."""
    global inference_engine
    
    try:
        # Try to find model checkpoint
        model_paths = [
            "models/checkpoints/best_model.pth",
            "models/best_model.pth",
            "checkpoints/best_model.pth",
            "best_model.pth"
        ]
        
        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = path
                break
        
        if model_path is None:
            logger.warning("No trained model found - web interface will start but inference will be disabled")
            return True  # Allow web interface to start even without model
        
        inference_engine = FlowerInferenceEngine(model_path, device="cpu")
        logger.info("Inference engine initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        return False

@app.route('/')
def index():
    """Main page with upload interface."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction."""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Secure filename
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            
            # Save file
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Run prediction
            if inference_engine is None:
                flash('Model not loaded. Please contact administrator.')
                return redirect(url_for('index'))
            
            result = inference_engine.predict_single(filepath)
            
            if 'error' in result:
                flash(f'Prediction error: {result["error"]}')
                return redirect(url_for('index'))
            
            # Prepare result data
            result_data = {
                'filename': filename,
                'filepath': filepath,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'inference_time_ms': result['inference_time_ms'],
                'timestamp': result['timestamp']
            }
            
            return render_template('result.html', result=result_data)
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload an image file.')
        return redirect(url_for('index'))

@app.route('/batch', methods=['POST'])
def batch_upload():
    """Handle batch file upload and prediction."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        if inference_engine is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Save files temporarily
        temp_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                temp_paths.append(filepath)
        
        if not temp_paths:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        # Run batch prediction
        results = inference_engine.predict_batch(temp_paths)
        
        # Prepare response
        response_data = {
            'total_files': len(temp_paths),
            'predictions': []
        }
        
        for i, (filepath, result) in enumerate(zip(temp_paths, results)):
            if 'error' not in result:
                response_data['predictions'].append({
                    'filename': os.path.basename(filepath),
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'inference_time_ms': result.get('inference_time_ms', 0)
                })
            else:
                response_data['predictions'].append({
                    'filename': os.path.basename(filepath),
                    'error': result['error']
                })
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single image prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        if inference_engine is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Run prediction
        result = inference_engine.predict_single(filepath)
        
        # Clean up file
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in API prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Display performance statistics."""
    if inference_engine is None:
        flash('Model not loaded')
        return redirect(url_for('index'))
    
    stats = inference_engine.get_performance_stats()
    return render_template('stats.html', stats=stats)

@app.route('/health')
def health():
    """Health check endpoint."""
    if inference_engine is None:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 503
    
    stats = inference_engine.get_performance_stats()
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'performance_stats': stats,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Initialize inference engine
    if init_inference_engine():
        logger.info("Starting web interface...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize inference engine. Exiting.")
