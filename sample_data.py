import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_financial_dataset(n_samples: int = 500, anomaly_ratio: float = 0.05) -> pd.DataFrame:
    """Generates synthetic financial transaction records with injected fraud/anomaly patterns."""
    np.random.seed(42)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # Normal transactions
    normal_amounts = np.random.gamma(shape=2.0, scale=40.0, size=n_normal) + 5.0
    normal_account_age = np.random.normal(loc=700, scale=250, size=n_normal).clip(10, 2000)
    normal_tx_count = np.random.poisson(lam=4, size=n_normal).clip(1, 25)
    normal_risk_score = np.random.beta(a=2, b=10, size=n_normal) * 100
    normal_distance = np.random.exponential(scale=15.0, size=n_normal).clip(0.1, 80.0)

    # Injected Anomalies (burst amounts, high velocity, high distance, new accounts)
    anom_amounts = np.random.uniform(3500.0, 18000.0, size=n_anomalies)
    anom_account_age = np.random.uniform(1.0, 15.0, size=n_anomalies) # very new accounts
    anom_tx_count = np.random.uniform(30.0, 95.0, size=n_anomalies) # burst transactions
    anom_risk_score = np.random.uniform(75.0, 99.5, size=n_anomalies)
    anom_distance = np.random.uniform(800.0, 8500.0, size=n_anomalies) # cross-continent jumps

    amounts = np.concatenate([normal_amounts, anom_amounts])
    account_age = np.concatenate([normal_account_age, anom_account_age])
    tx_count = np.concatenate([normal_tx_count, anom_tx_count])
    risk_score = np.concatenate([normal_risk_score, anom_risk_score])
    distance = np.concatenate([normal_distance, anom_distance])

    start_date = datetime.now() - timedelta(days=30)
    timestamps = [start_date + timedelta(minutes=int(i * (30*24*60 / n_samples))) for i in range(n_samples)]

    df = pd.DataFrame({
        "TransactionID": [f"TX-{100000 + i}" for i in range(n_samples)],
        "Timestamp": timestamps,
        "Amount": np.round(amounts, 2),
        "AccountAgeDays": np.round(account_age).astype(int),
        "DailyTransactionCount": np.round(tx_count).astype(int),
        "RiskScore": np.round(risk_score, 2),
        "LocationDistanceKM": np.round(distance, 1)
    })

    # Shuffle rows
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)

def generate_server_metrics_dataset(n_samples: int = 500, anomaly_ratio: float = 0.05) -> pd.DataFrame:
    """Generates synthetic server telemetry data with CPU, memory, and latency spikes."""
    np.random.seed(99)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # Normal metrics
    normal_cpu = np.random.normal(loc=35, scale=10, size=n_normal).clip(5, 70)
    normal_mem = np.random.normal(loc=55, scale=8, size=n_normal).clip(20, 75)
    normal_network = np.random.gamma(shape=3.0, scale=12.0, size=n_normal)
    normal_latency = np.random.normal(loc=18, scale=5, size=n_normal).clip(2, 45)
    normal_errors = np.random.poisson(lam=0.2, size=n_normal)

    # Anomaly metrics
    anom_cpu = np.random.uniform(94, 100, size=n_anomalies)
    anom_mem = np.random.uniform(92, 99.8, size=n_anomalies)
    anom_network = np.random.uniform(250, 900, size=n_anomalies)
    anom_latency = np.random.uniform(300, 1500, size=n_anomalies)
    anom_errors = np.random.uniform(20, 85, size=n_anomalies).astype(int)

    cpu = np.concatenate([normal_cpu, anom_cpu])
    mem = np.concatenate([normal_mem, anom_mem])
    network = np.concatenate([normal_network, anom_network])
    latency = np.concatenate([normal_latency, anom_latency])
    errors = np.concatenate([normal_errors, anom_errors])

    start_time = datetime.now() - timedelta(hours=12)
    timestamps = [start_time + timedelta(seconds=int(i * (12*3600 / n_samples))) for i in range(n_samples)]

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "CPU_Usage_Pct": np.round(cpu, 1),
        "Memory_Usage_Pct": np.round(mem, 1),
        "Network_IO_MB": np.round(network, 2),
        "Disk_Latency_ms": np.round(latency, 1),
        "Error_Count": errors
    })

    return df.sample(frac=1.0, random_state=99).reset_index(drop=True)

def create_sample_files(target_dir: str):
    """Saves sample CSV files to disk."""
    os.makedirs(target_dir, exist_ok=True)
    df_fin = generate_financial_dataset(400)
    df_srv = generate_server_metrics_dataset(400)
    
    fin_path = os.path.join(target_dir, "sample_financial_transactions.csv")
    srv_path = os.path.join(target_dir, "sample_server_metrics.csv")
    
    df_fin.to_csv(fin_path, index=False)
    df_srv.to_csv(srv_path, index=False)
    return fin_path, srv_path
