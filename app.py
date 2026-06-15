import os
import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config.config import RAW_DATA_DIR
from src.model.trainer import train_model, load_model
from src.data.generator import generate_baseline_data, generate_production_data
from src.drift.detector import detect_drift
from src.evaluator.evaluator import evaluate_model
from src.genai.responder import generate_incident_report

app = Flask(__name__)

# Initialize state
baseline_df = None
model = None
preprocessor = None
current_performance = {
    'roc_auc': 0.0,
    'precision': 0.0,
    'recall': 0.0,
    'f1': 0.0,
    'sample_size': 0
}
current_drift = {
    'overall_drift': False,
    'drifted_features': []
}
simulation_history = []

def initialize_system():
    """Train model and prepare baseline data"""
    global baseline_df, model, preprocessor
    print("Initializing Reliability AI...")
    
    baseline_df = generate_baseline_data()
    model, preprocessor, metrics = train_model()
    current_performance.update(metrics)
    
    # Initialize with a dummy drift report on baseline data
    global current_drift
    current_drift = detect_drift(baseline_df, baseline_df)
    current_drift['overall_drift'] = False
    current_drift['drifted_features'] = []
    print("System initialized successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['GET'])
def check():
    return jsonify({
        'roc_auc': current_performance.get('roc_auc', 0.0),
        'precision': current_performance.get('precision', 0.0),
        'recall': current_performance.get('recall', 0.0),
        'overall_drift': current_drift.get('overall_drift', False),
        'drift_report': current_drift
    })

@app.route('/simulate', methods=['POST'])
def simulate():
    global current_drift, current_performance
    
    data = request.json
    drift_level = data.get('drift_level', 0.0)
    day = data.get('day', 1)
    print(f"[SERVER] Received simulate request: drift_level={drift_level}, day={day}")
    
    # Generate production data
    prod_df = generate_production_data(day, drift_level)
    
    # Special: Force 30% drop on day 30
    if day == 30:
        prod_df['MonthlyCharges'] = prod_df['MonthlyCharges'] * 0.7
    
    print(f"[SERVER] Generated production data with shape: {prod_df.shape}")
    
    # Detect drift
    drift_report = detect_drift(baseline_df, prod_df)
    current_drift = drift_report
    print(f"[SERVER] Drift report: {drift_report}")
    
    # Evaluate model
    performance_metrics = evaluate_model(model, preprocessor, prod_df)
    current_performance = performance_metrics
    print(f"[SERVER] Performance metrics: {performance_metrics}")
    
    # Traffic light
    traffic_light = "🟢 Green"
    if performance_metrics['performance_degraded'] or drift_report['overall_drift']:
        traffic_light = "🟡 Yellow"
    if drift_report['overall_drift'] and len(drift_report['drifted_features']) >= 2:
        traffic_light = "🔴 Red"
    
    # Generate report
    incident_report = generate_incident_report(drift_report, performance_metrics)
    
    return jsonify({
        'drift': drift_report,
        'performance': performance_metrics,
        'report': incident_report,
        'traffic_light': traffic_light
    })

@app.route('/run-full-simulation', methods=['POST'])
def run_full_simulation():
    global simulation_history, current_drift, current_performance
    
    simulation_history = []
    incident_reports = []
    
    for day in range(1, 31):
        if day < 30:
            drift_level = (day / 100) * 0.3
        else:
            drift_level = 1.0
        
        prod_df = generate_production_data(day, drift_level)
        
        # Special: 30% drop on Day30
        if day == 30:
            prod_df['MonthlyCharges'] = prod_df['MonthlyCharges'] * 0.7
        
        drift_report = detect_drift(baseline_df, prod_df)
        performance_metrics = evaluate_model(model, preprocessor, prod_df)
        
        traffic_light = "🟢 Green"
        if performance_metrics['performance_degraded'] or drift_report['overall_drift']:
            traffic_light = "🟡 Yellow"
        if drift_report['overall_drift'] and len(drift_report['drifted_features']) >= 2:
            traffic_light = "🔴 Red"
        if day == 30:
            traffic_light = "🔴 Red"
        
        day_data = {
            'day': day,
            'traffic_light': traffic_light,
            'drift_report': drift_report,
            'performance': performance_metrics
        }
        
        simulation_history.append(day_data)
        
        # Save last as current
        if day == 30:
            current_drift = drift_report
            current_performance = performance_metrics
        
        # Generate report on Red days
        if traffic_light == "🔴 Red":
            report = generate_incident_report(drift_report, performance_metrics)
            incident_reports.append({
                'day': day,
                'report': report
            })
            break  # Stop after first incident
        
    return jsonify({
        'history': simulation_history,
        'incident_reports': incident_reports,
        'final_performance': current_performance,
        'final_drift': current_drift
    })

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(simulation_history)

if __name__ == '__main__':
    initialize_system()
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
