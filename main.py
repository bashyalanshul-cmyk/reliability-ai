
#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, TARGET_COLUMN
from src.data.generator import generate_baseline_data, generate_production_data
from src.model.trainer import preprocess_data, train_model
from src.drift.detector import detect_drift
from src.evaluator.evaluator import evaluate_model
from src.genai.responder import generate_incident_report

def main():
    print("\n" + "="*80)
    print("🚀 RELIABILITY AI - AUTONOMOUS MODEL MONITORING")
    print("="*80)

    # Step 1: The Baseline (The "Normal" Pulse)
    print("\n📊 Step 1: Establishing Baseline")
    print("-"*80)
    baseline_df = generate_baseline_data()
    model, preprocessor, baseline_metrics = train_model()
    print(f"✅ Baseline established!")
    print(f"   - ROC-AUC: {baseline_metrics['roc_auc']:.3f}")
    print(f"   - Precision: {baseline_metrics['precision']:.3f}")
    print(f"   - Recall: {baseline_metrics['recall']:.3f}")

    # Step 2 & 3: The Production Simulator (The "Drift" Injection) & Traffic Light
    print("\n🎬 Step 2 & 3: Running Production Simulation")
    print("-"*80)
    
    simulation_history = []
    
    for day in range(1, 31):  # 30 days
        # Special: Inject 30% drop in MonthlyCharges on Day 30!
        if day < 30:
            drift_level = (day / 100) * 0.3  # Gradual drift up to ~10% at day 30
        else:
            drift_level = 1.0  # MAX DRIFT ON DAY 30!
        
        print(f"\n📅 Day {day:02d}")
        
        # Generate production data
        prod_df = generate_production_data(day, drift_level)
        
        # Special Day 30: Force 30% drop in MonthlyCharges!
        if day == 30:
            print("⚠️  DRIFT INJECTION: Dropping MonthlyCharges by 30%!")
            prod_df['MonthlyCharges'] = prod_df['MonthlyCharges'] * 0.7
        
        # Step 3: The Statistical Traffic Light
        drift_report = detect_drift(baseline_df, prod_df)
        performance_metrics = evaluate_model(model, preprocessor, prod_df)
        
        # Traffic light logic
        traffic_light = "🟢 Green"
        drift_severity = "LOW"
        
        if performance_metrics['performance_degraded'] or drift_report['overall_drift']:
            traffic_light = "🟡 Yellow"
            drift_severity = "MEDIUM"
            
        if drift_report['overall_drift'] and len(drift_report['drifted_features']) >= 2:
            traffic_light = "🔴 Red"
            drift_severity = "HIGH"
            
        if day == 30:
            traffic_light = "🔴 Red"
            drift_severity = "CRITICAL"
        
        print(f"   {traffic_light} - {drift_severity}")
        print(f"   - ROC-AUC: {performance_metrics['roc_auc']:.3f}")
        print(f"   - Drifted features: {', '.join(drift_report['drifted_features']) if drift_report['drifted_features'] else 'None'}")
        
        # Save history
        simulation_history.append({
            'day': day,
            'traffic_light': traffic_light,
            'drift_severity': drift_severity,
            'drift_report': drift_report,
            'performance': performance_metrics,
            'drift_level': drift_level
        })
        
        # Step 4: Autonomous Incident Responder on Red
        if traffic_light == "🔴 Red":
            print("\n🔴 ALERT! TRIGGERING INCIDENT RESPONDER!")
            incident_report = generate_incident_report(drift_report, performance_metrics)
            print("\n📄 GENERATED INCIDENT REPORT:")
            print("-"*80)
            print(incident_report)
            
            # Save report
            report_dir = PROCESSED_DATA_DIR / "reports"
            report_dir.mkdir(exist_ok=True)
            report_path = report_dir / f"incident_report_day_{day:02d}.txt"
            with open(report_path, "w") as f:
                f.write(incident_report)
            print(f"\n✅ Report saved to: {report_path}")
            break  # Stop after incident for demo
    
    print("\n" + "="*80)
    print("✅ SIMULATION COMPLETE!")
    print("="*80)
    
    # Save simulation history
    history_path = PROCESSED_DATA_DIR / "simulation_history.joblib"
    joblib.dump(simulation_history, history_path)
    print(f"\n📊 History saved to: {history_path}")

if __name__ == "__main__":
    main()
