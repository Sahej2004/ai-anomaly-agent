import sys
from sample_data import generate_financial_dataset, generate_server_metrics_dataset
from detector import AnomalyDetector
from ai_agent import AnomalyAIAgent
from email_alerter import EmailAlerter

def run_tests():
    print("=== 1. Testing Sample Data Generation ===")
    df_fin = generate_financial_dataset(300, anomaly_ratio=0.06)
    print(f"Generated Financial Dataset: {df_fin.shape[0]} rows, {df_fin.shape[1]} columns")
    assert len(df_fin) == 300, "Dataset row count mismatch"

    print("\n=== 2. Testing Anomaly Detector (Isolation Forest) ===")
    detector = AnomalyDetector(method="Isolation Forest", contamination=0.06)
    results = detector.detect(df_fin)
    summary = results["summary_metrics"]
    anomaly_df = results["anomaly_df"]
    
    print(f"Detection Success! Total: {summary['total_records']}, Anomalies: {summary['anomaly_count']} ({summary['anomaly_percentage']}%)")
    print(f"Critical Anomalies: {summary['critical_count']}")
    print(f"Top Affected Columns: {summary['top_affected_columns']}")
    assert summary["anomaly_count"] > 0, "No anomalies detected!"
    assert "anomaly_score" in anomaly_df.columns, "Missing anomaly_score column"
    assert "severity" in anomaly_df.columns, "Missing severity column"

    print("\n=== 3. Testing Anomaly Detector (LOF) ===")
    detector_lof = AnomalyDetector(method="Local Outlier Factor (LOF)", contamination=0.05)
    results_lof = detector_lof.detect(df_fin)
    print(f"LOF Success! Anomalies: {results_lof['summary_metrics']['anomaly_count']}")

    print("\n=== 4. Testing AI Agent Summary Generation ===")
    agent = AnomalyAIAgent() # Will use fallback or configured API key
    ai_output = agent.generate_summary(summary, anomaly_df.head(5).to_dict(orient="records"))
    print(f"AI Provider Source: {ai_output.get('source')}")
    print(f"Email Subject: {ai_output.get('email_subject')}")
    assert len(ai_output.get("executive_summary", "")) > 50, "Executive summary too short or empty"
    assert len(ai_output.get("recommendations", "")) > 30, "Recommendations missing"

    print("\n=== 5. Testing Email HTML Generator & CSV Export ===")
    alerter = EmailAlerter()
    html_report = alerter.build_html_report(summary, ai_output, anomaly_df)
    print(f"Generated HTML report length: {len(html_report)} chars")
    assert "AI Anomaly Detection Report" in html_report, "HTML report header missing"
    assert str(summary["anomaly_count"]) in html_report, "Anomaly count not found in HTML"

    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY! The pipeline is fully operational.")

if __name__ == "__main__":
    run_tests()
