import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple, List

class AnomalyDetector:
    """
    Advanced multi-algorithm anomaly detector with feature attribution.
    """
    def __init__(self, method: str = "Isolation Forest", contamination: float = 0.05):
        self.method = method
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.numeric_cols: List[str] = []
        self.feature_names: List[str] = []

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Preprocesses dataframe: selects numeric features, imputes missing values, and scales.
        """
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found in the dataset for anomaly detection.")
        
        # Drop columns with 100% NaN
        valid_cols = [c for c in numeric_df.columns if numeric_df[c].notna().sum() > 0]
        if not valid_cols:
            raise ValueError("All numeric columns are empty or NaN.")
            
        self.numeric_cols = valid_cols
        self.feature_names = valid_cols
        
        # Impute missing values
        imputed_data = self.imputer.fit_transform(numeric_df[valid_cols])
        # Scale features
        scaled_data = self.scaler.fit_transform(imputed_data)
        
        return scaled_data, numeric_df[valid_cols]

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs anomaly detection on the provided dataframe.
        Returns enriched dataframe with anomaly labels, scores, and statistics.
        """
        original_df = df.copy()
        scaled_data, numeric_df = self.preprocess(original_df)
        n_samples = scaled_data.shape[0]

        if n_samples < 5:
            raise ValueError("Dataset is too small for anomaly detection (requires at least 5 rows).")

        # Select algorithm
        if self.method == "Isolation Forest":
            clf = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=150
            )
            raw_preds = clf.fit_predict(scaled_data) # -1 for anomaly, 1 for normal
            is_anomaly = (raw_preds == -1)
            # decision_function gives negative scores for outliers (lower is more anomalous)
            raw_scores = -clf.decision_function(scaled_data)
            # Normalize scores to 0.0 - 1.0 (higher = more anomalous)
            min_s, max_s = raw_scores.min(), raw_scores.max()
            if max_s > min_s:
                normalized_scores = (raw_scores - min_s) / (max_s - min_s)
            else:
                normalized_scores = np.zeros(n_samples)

        elif self.method == "Local Outlier Factor (LOF)":
            n_neighbors = min(20, max(2, n_samples - 1))
            clf = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=self.contamination,
                novelty=False
            )
            raw_preds = clf.fit_predict(scaled_data)
            is_anomaly = (raw_preds == -1)
            # negative_outlier_factor_: closer to -1 means inlier, large negative means outlier
            raw_scores = -clf.negative_outlier_factor_
            min_s, max_s = raw_scores.min(), raw_scores.max()
            if max_s > min_s:
                normalized_scores = (raw_scores - min_s) / (max_s - min_s)
            else:
                normalized_scores = np.zeros(n_samples)

        elif self.method == "Z-Score (Statistical)":
            # Multivariate Z-score thresholding
            z_scores = np.abs(scaled_data)
            max_z_per_row = np.max(z_scores, axis=1)
            # Dynamic threshold based on contamination quantile
            thresh = np.quantile(max_z_per_row, 1.0 - self.contamination)
            is_anomaly = max_z_per_row >= thresh
            min_s, max_s = max_z_per_row.min(), max_z_per_row.max()
            normalized_scores = (max_z_per_row - min_s) / (max_s - min_s) if max_s > min_s else np.zeros(n_samples)

        else: # Ensemble (Isolation Forest + Z-Score)
            clf = IsolationForest(contamination=self.contamination, random_state=42)
            if_preds = clf.fit_predict(scaled_data) == -1
            z_scores = np.abs(scaled_data)
            z_preds = np.max(z_scores, axis=1) > 2.5
            is_anomaly = np.logical_or(if_preds, z_preds)
            raw_scores = -clf.decision_function(scaled_data)
            min_s, max_s = raw_scores.min(), raw_scores.max()
            normalized_scores = (raw_scores - min_s) / (max_s - min_s) if max_s > min_s else np.zeros(n_samples)

        # Compute feature attribution (which feature deviates most for each anomaly)
        abs_deviations = np.abs(scaled_data)
        top_driver_indices = np.argmax(abs_deviations, axis=1)
        top_drivers = [self.feature_names[idx] for idx in top_driver_indices]

        # Classify severity
        severities = []
        for is_a, score in zip(is_anomaly, normalized_scores):
            if not is_a:
                severities.append("Normal")
            elif score > 0.8:
                severities.append("High / Critical")
            elif score > 0.5:
                severities.append("Medium")
            else:
                severities.append("Low")

        # Append results to original DataFrame
        result_df = original_df.copy()
        result_df["is_anomaly"] = is_anomaly
        result_df["anomaly_score"] = np.round(normalized_scores, 4)
        result_df["severity"] = severities
        result_df["primary_anomaly_factor"] = top_drivers

        # Summary statistics
        total_records = len(result_df)
        anomaly_count = int(is_anomaly.sum())
        anomaly_pct = round((anomaly_count / total_records) * 100, 2)
        critical_count = int((result_df["severity"] == "High / Critical").sum())

        # Top anomalous features overall
        anomaly_rows = result_df[result_df["is_anomaly"]]
        top_affected_columns = anomaly_rows["primary_anomaly_factor"].value_counts().to_dict() if not anomaly_rows.empty else {}

        # Column-level summary (mean difference between normal and anomaly)
        column_impact = {}
        if anomaly_count > 0 and len(result_df) > anomaly_count:
            normal_rows = result_df[~result_df["is_anomaly"]]
            for col in self.numeric_cols:
                normal_mean = normal_rows[col].mean()
                anomaly_mean = anomaly_rows[col].mean()
                diff_pct = 0.0
                if normal_mean != 0 and pd.notna(normal_mean) and pd.notna(anomaly_mean):
                    diff_pct = round(((anomaly_mean - normal_mean) / abs(normal_mean)) * 100, 2)
                column_impact[col] = {
                    "normal_mean": round(float(normal_mean), 2) if pd.notna(normal_mean) else 0,
                    "anomaly_mean": round(float(anomaly_mean), 2) if pd.notna(anomaly_mean) else 0,
                    "deviation_pct": diff_pct
                }

        summary_metrics = {
            "total_records": total_records,
            "anomaly_count": anomaly_count,
            "normal_count": total_records - anomaly_count,
            "anomaly_percentage": anomaly_pct,
            "critical_count": critical_count,
            "method_used": self.method,
            "contamination_rate": self.contamination,
            "numeric_columns": self.numeric_cols,
            "top_affected_columns": top_affected_columns,
            "column_impact": column_impact
        }

        return {
            "result_df": result_df,
            "anomaly_df": result_df[result_df["is_anomaly"]].sort_values(by="anomaly_score", ascending=False),
            "summary_metrics": summary_metrics,
            "scaled_data": scaled_data
        }
