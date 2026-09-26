import os
import json
import re
from typing import Dict, Any, Optional, Tuple

# Try new google-genai SDK first, then legacy google.generativeai
HAS_NEW_GENAI = False
HAS_LEGACY_GENAI = False

try:
    from google import genai
    HAS_NEW_GENAI = True
except ImportError:
    pass

if not HAS_NEW_GENAI:
    try:
        import google.generativeai as legacy_genai
        HAS_LEGACY_GENAI = True
    except ImportError:
        pass


class AnomalyAIAgent:
    """
    AI Agent that analyzes anomaly detection results using Google Gemini,
    extracts actionable insights, and crafts professional alert notifications.

    Only text-generation Gemini models are accepted. TTS, audio, image,
    embedding, and live/audio models are intentionally excluded.
    """

    # Explicit text-generation models only.
    # Keeping this list avoids accidentally selecting TTS or other modalities
    # returned by the Google model-discovery API.
    SAFE_TEXT_MODELS = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    @staticmethod
    def _is_text_generation_model(model_name: str) -> bool:
        """Reject models intended for TTS/audio/image/embedding/live workloads."""
        name = model_name.lower().replace("models/", "")

        blocked_terms = (
            "tts",
            "audio",
            "native-audio",
            "image",
            "embedding",
            "embed",
            "live",
            "robotics",
        )

        if any(term in name for term in blocked_terms):
            return False

        # Only use known Gemini text-generation model families.
        return name.startswith("gemini-")

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash"
    ):
        # Check explicit parameter, then Streamlit secrets, then environment variables.
        st_key = None

        try:
            import streamlit as st

            if "GEMINI_API_KEY" in st.secrets:
                st_key = str(st.secrets["GEMINI_API_KEY"]).strip()
            elif "GOOGLE_API_KEY" in st.secrets:
                st_key = str(st.secrets["GOOGLE_API_KEY"]).strip()
        except Exception:
            pass

        self.api_key = (
            api_key
            or st_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        requested_model = (model_name or "gemini-2.5-flash").strip()
        if not self._is_text_generation_model(requested_model):
            requested_model = "gemini-2.5-flash"

        self.model_name = requested_model
        self.client = None
        self.client_ready = False
        self.init_error = None

        if self.api_key:
            if HAS_NEW_GENAI:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                    self.client_ready = True
                except Exception as e:
                    self.init_error = f"google-genai client init error: {e}"

            if not self.client_ready and HAS_LEGACY_GENAI:
                try:
                    legacy_genai.configure(api_key=self.api_key)
                    self.client_ready = True
                except Exception as e:
                    self.init_error = f"google.generativeai configure error: {e}"
        else:
            self.init_error = "No API key configured"

    @staticmethod
    def list_available_models(api_key: str) -> Tuple[bool, list, str]:
        """
        Lists only models that are appropriate for normal text generation.

        TTS/audio/image/embedding/live models are deliberately filtered out.
        """
        if not api_key:
            return False, [], "No API key provided."

        found = []

        # Legacy SDK model discovery
        if HAS_LEGACY_GENAI:
            try:
                legacy_genai.configure(api_key=api_key)

                for m in legacy_genai.list_models():
                    name = getattr(m, "name", "").replace("models/", "")
                    methods = getattr(m, "supported_generation_methods", [])

                    if (
                        "generateContent" in methods
                        and AnomalyAIAgent._is_text_generation_model(name)
                    ):
                        if name not in found:
                            found.append(name)

            except Exception:
                pass

        # New SDK model discovery
        if HAS_NEW_GENAI:
            try:
                client = genai.Client(api_key=api_key)

                for m in client.models.list():
                    name = getattr(m, "name", str(m)).replace("models/", "")

                    if AnomalyAIAgent._is_text_generation_model(name):
                        if name not in found:
                            found.append(name)

            except Exception:
                pass

        if found:
            # Put commonly useful text models first.
            ordered = []

            for safe_model in AnomalyAIAgent.SAFE_TEXT_MODELS:
                if safe_model in found and safe_model not in ordered:
                    ordered.append(safe_model)

            for model in found:
                if model not in ordered:
                    ordered.append(model)

            return True, ordered, f"Found {len(ordered)} text-generation models."

        return (
            False,
            [],
            "No compatible text-generation Gemini models were found. "
            "Check the API key and Gemini API access."
        )

    def generate_summary(
        self,
        summary_metrics: Dict[str, Any],
        sample_anomalies: list
    ) -> Dict[str, str]:
        """Generate comprehensive AI analysis and alert content."""

        if self.client_ready:
            try:
                return self._call_gemini_analysis(
                    summary_metrics,
                    sample_anomalies
                )
            except Exception as e:
                return self._generate_fallback_summary(
                    summary_metrics,
                    sample_anomalies,
                    error_msg=str(e)
                )

        return self._generate_fallback_summary(
            summary_metrics,
            sample_anomalies,
            error_msg=self.init_error if self.api_key else None
        )

    def _call_gemini_analysis(
        self,
        summary_metrics: Dict[str, Any],
        sample_anomalies: list
    ) -> Dict[str, str]:

        prompt = f"""
You are an expert Autonomous AI Data Quality & Anomaly Detection Agent.

You have just analyzed a dataset and detected statistical and machine-learning
anomalies.

Here is the statistical summary of the detection:
{json.dumps(summary_metrics, indent=2)}

Here are sample detected anomalous records (up to 5 most severe):
{json.dumps(sample_anomalies[:5], indent=2, default=str)}

Respond with a JSON object ONLY. Do not use markdown fences.

Required JSON structure:
{{
  "executive_summary": "High-level 2-3 paragraph executive summary of findings, data health, and risk assessment.",
  "root_causes": "Bullet points detailing likely root causes and which specific features contributed most to the anomalies.",
  "recommendations": "Actionable, numbered list of recommendations for the operations/engineering/business team.",
  "email_subject": "A concise, high-priority email alert subject line.",
  "email_body": "A professionally formatted plain text or markdown email body ready to be sent to stakeholders."
}}
"""

        # IMPORTANT:
        # Do NOT append every model returned by Google.
        # That previous behavior could include TTS/audio/image models.
        candidate_models = []

        if self._is_text_generation_model(self.model_name):
            candidate_models.append(self.model_name)

        # Add only known safe text models as fallbacks.
        for model in self.SAFE_TEXT_MODELS:
            if model not in candidate_models:
                candidate_models.append(model)

        last_error = None

        for model_to_try in candidate_models:
            try:
                text = ""

                # Modern google-genai SDK
                if HAS_NEW_GENAI and self.client:
                    response = self.client.models.generate_content(
                        model=model_to_try,
                        contents=prompt
                    )

                    text = (response.text or "").strip()

                # Legacy google-generativeai SDK
                elif HAS_LEGACY_GENAI:
                    model = legacy_genai.GenerativeModel(model_to_try)
                    response = model.generate_content(prompt)
                    text = (response.text or "").strip()

                else:
                    raise RuntimeError("No Gemini SDK available.")

                if not text:
                    raise RuntimeError(
                        f"Gemini returned an empty response for {model_to_try}."
                    )

                # Robust JSON extraction
                parsed = None
                json_match = re.search(r"\{[\s\S]*\}", text)

                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                    except Exception:
                        parsed = None

                if parsed and isinstance(parsed, dict) and "executive_summary" in parsed:
                    return {
                        "executive_summary": parsed.get("executive_summary", ""),
                        "root_causes": parsed.get("root_causes", ""),
                        "recommendations": parsed.get("recommendations", ""),
                        "email_subject": parsed.get(
                            "email_subject",
                            f"🚨 Anomaly Alert: {summary_metrics['anomaly_count']} outliers detected"
                        ),
                        "email_body": parsed.get("email_body", ""),
                        "source": f"Google Gemini ({model_to_try})",
                        "error_details": None
                    }

                # If Gemini returned normal text instead of JSON
                return {
                    "executive_summary": text,
                    "root_causes": "See detailed breakdown in the executive summary above.",
                    "recommendations": "Investigate flagged records with high anomaly scores.",
                    "email_subject": (
                        f"🚨 Anomaly Alert: "
                        f"{summary_metrics['anomaly_count']} anomalies detected "
                        f"({summary_metrics['anomaly_percentage']}%)"
                    ),
                    "email_body": text,
                    "source": f"Google Gemini ({model_to_try})",
                    "error_details": None
                }

            except Exception as e:
                err_str = str(e)
                last_error = e

                # Model-specific failure: try the next SAFE text model.
                if (
                    "404" in err_str
                    or "NOT_FOUND" in err_str
                    or "not found" in err_str.lower()
                ):
                    continue

                # Quota/auth/permission problems should be shown to the user.
                raise e

        if last_error:
            raise last_error

        raise RuntimeError("No compatible Gemini text model was available.")

    def _generate_fallback_summary(
        self,
        summary_metrics: Dict[str, Any],
        sample_anomalies: list,
        error_msg: Optional[str] = None
    ) -> Dict[str, str]:

        total = summary_metrics.get("total_records", 0)
        count = summary_metrics.get("anomaly_count", 0)
        pct = summary_metrics.get("anomaly_percentage", 0.0)
        critical = summary_metrics.get("critical_count", 0)
        method = summary_metrics.get("method_used", "Statistical ML")
        top_factors = summary_metrics.get("top_affected_columns", {})

        top_factors_str = (
            ", ".join(
                [
                    f"{k} ({v} occurrences)"
                    for k, v in list(top_factors.items())[:3]
                ]
            )
            if top_factors
            else "various numeric attributes"
        )

        if pct > 10 or critical > 5:
            risk_level = "CRITICAL / HIGH"
            risk_note = (
                "Elevated outlier density indicates potential systemic malfunction, "
                "fraud surge, or data collection integrity breakdown."
            )
        elif pct > 4:
            risk_level = "MODERATE"
            risk_note = (
                "Moderate anomalies detected. Notable variance from standard "
                "baseline operating metrics."
            )
        else:
            risk_level = "LOW / NORMAL FLUCTUATIONS"
            risk_note = (
                "Anomalies are isolated and represent standard tail-distribution variance."
            )

        first_factor = (
            list(top_factors.keys())[0]
            if top_factors
            else "feature distribution"
        )

        exec_summary = (
            f"### Executive Anomaly Brief\n\n"
            f"- **Dataset Scope**: {total:,} total records evaluated using **{method}**.\n"
            f"- **Outliers Identified**: **{count:,} records ({pct}%)** met outlier criteria.\n"
            f"- **Critical Outliers**: **{critical:,} records** scored in the high-severity tier.\n"
            f"- **Overall Risk Level**: **{risk_level}**.\n\n"
            f"{risk_note}\n\n"
            f"The primary variance drivers identified across the flagged records were: "
            f"**{top_factors_str}**."
        )

        root_causes = (
            f"- **Key Deviation Factor**: The most frequent driver for flagged anomalies is `{first_factor}`.\n"
            f"- **Multivariate Shifts**: Significant cross-feature deviation observed between normal baseline averages and anomaly cohorts.\n"
            f"- **Potential Etiologies**: Sensor drift, fraudulent/unusual transaction spikes, extreme user behavior, or input formatting discrepancies."
        )

        recommendations = (
            f"1. **Triage Critical Rows**: Immediately inspect the top {min(critical, 10)} records with anomaly scores > 0.80.\n"
            f"2. **Feature Deep-Dive**: Validate integrity and pipeline ingestion for primary driver: `{first_factor}`.\n"
            f"3. **Stakeholder Notification**: Dispatch the automated email alert to operational and engineering teams for follow-up.\n"
            f"4. **Threshold Calibration**: Adjust algorithm contamination settings if historical tolerance differs from {summary_metrics.get('contamination_rate', 0.05) * 100}%."
        )

        email_subj = (
            f"[{risk_level} ALERT] {count} Anomalies Detected "
            f"in Dataset ({pct}% Outlier Ratio)"
        )

        email_body = (
            "ANOMALY DETECTION SYSTEM ALERT\n"
            "=========================================\n\n"
            f"Detection Algorithm : {method}\n"
            f"Total Records       : {total:,}\n"
            f"Anomalies Found     : {count:,} ({pct}%)\n"
            f"Critical Outliers   : {critical:,}\n"
            f"Risk Assessment     : {risk_level}\n"
            f"Primary Drivers     : {top_factors_str}\n\n"
            "RECOMMENDED IMMEDIATE ACTIONS:\n"
            "1. Review the attached anomaly report and verify high-severity records.\n"
            f"2. Check data pipeline for {top_factors_str}.\n"
            "3. Acknowledge and resolve alerts in the operations dashboard.\n\n"
            "Generated automatically by AI Anomaly Detection Agent."
        )

        source_info = (
            "Built-in Rule Engine "
            "(Configure Gemini API Key in sidebar for AI synthesis)"
        )

        if error_msg:
            source_info += f" [Note: Gemini API returned: {error_msg}]"

        return {
            "executive_summary": exec_summary,
            "root_causes": root_causes,
            "recommendations": recommendations,
            "email_subject": email_subj,
            "email_body": email_body,
            "source": source_info,
            "error_details": error_msg
        }
