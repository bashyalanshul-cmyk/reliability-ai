
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def generate_pdf_report(report_text, output_path):
    """Generate a PDF report from text using ReportLab"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch
        
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        story.append(Paragraph("🛡️ Reliability AI - Incident Report", styles['Title']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Add report body
        for paragraph in report_text.split('\n\n'):
            if paragraph.strip():
                # Check if it's a heading (starts with #)
                if paragraph.strip().startswith('#'):
                    level = len([c for c in paragraph if c == '#'])
                    heading_text = paragraph.strip().lstrip('#').strip()
                    if level == 1:
                        story.append(Paragraph(heading_text, styles['Heading1']))
                    elif level == 2:
                        story.append(Paragraph(heading_text, styles['Heading2']))
                    else:
                        story.append(Paragraph(heading_text, styles['Heading3']))
                else:
                    story.append(Paragraph(paragraph.replace('\n', '<br/>'), styles['BodyText']))
                story.append(Spacer(1, 0.1 * inch))
        
        doc.build(story)
        print(f"✅ PDF report saved to: {output_path}")
        return True
    except Exception as e:
        print(f"⚠️  Could not generate PDF: {e}")
        return False

def generate_incident_report(drift_report, performance_metrics, save_pdf=True):
    """Generate GenAI incident report (fallback if no API key)"""
    api_key = os.getenv('OPENAI_API_KEY')
    report_text = None
    
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            prompt = f"""
            You are an MLOps incident response expert. Analyze the following drift and performance data:
            
            DRIFT REPORT:
            {json.dumps(drift_report, indent=2)}
            
            PERFORMANCE METRICS:
            {json.dumps(performance_metrics, indent=2)}
            
            Generate a comprehensive executive report with:
            1. Executive Summary (what happened)
            2. Root Cause Analysis (which features drifted and why)
            3. Business Impact Assessment
            4. Recommended Mitigation Strategy
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior MLOps engineer and incident responder."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            report_text = response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            report_text = None
    
    # Fallback to built-in report
    if not report_text:
        report_text = f"""
# RELIABILITY AI INCIDENT REPORT
## Executive Summary
- {'DATA DRIFT DETECTED' if drift_report['overall_drift'] else 'No data drift'}
- {'PERFORMANCE DEGRADED' if performance_metrics.get('performance_degraded', False) else 'Performance stable'}
- Drifted features: {', '.join(drift_report['drifted_features'])}

## Root Cause Analysis
- Drift detected in features: {', '.join(drift_report['drifted_features'])}
- Performance metrics:
  - ROC-AUC: {performance_metrics.get('roc_auc', 'N/A'):.3f}
  - Precision: {performance_metrics.get('precision', 'N/A'):.3f}
  - Recall: {performance_metrics.get('recall', 'N/A'):.3f}

## Business Impact
- Customer churn predictions may be less accurate
- Potential revenue impact from missed churn cases

## Recommended Mitigation
1. Retrain model on fresh data
2. Monitor drifted features closely
3. Validate data collection pipeline
        """
    
    # Save PDF if requested
    if save_pdf:
        from config.config import PROCESSED_DATA_DIR
        report_dir = PROCESSED_DATA_DIR / "reports"
        report_dir.mkdir(exist_ok=True)
        pdf_path = report_dir / "incident_report.pdf"
        generate_pdf_report(report_text, pdf_path)
    
    return report_text
