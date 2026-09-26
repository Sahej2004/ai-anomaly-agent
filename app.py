import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import io
from dotenv import load_dotenv

from detector import AnomalyDetector
from ai_agent import AnomalyAIAgent
from email_alerter import EmailAlerter
from sample_data import generate_financial_dataset, generate_server_metrics_dataset

load_dotenv()

# Page setup
st.set_page_config(
    page_title="AI Anomaly Detection Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "detection_results" not in st.session_state:
    st.session_state.detection_results = None
if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = None
if "email_log" not in st.session_state:
    st.session_state.email_log = []

# Helper to read credentials from st.secrets or os.getenv
def get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return os.getenv(key, default)

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Agent Settings")
    
    with st.expander("🤖 Gemini AI Configuration", expanded=True):
        default_gemini_key = get_secret("GEMINI_API_KEY", "")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=default_gemini_key,
            type="password",
            help="Get your free API key from Google AI Studio (aistudio.google.com)"
        )
        gemini_model = st.selectbox(
            "Model Selection",
            ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            index=0
        )
        if gemini_api_key:
            st.success("API Key detected")
            if st.button("🧪 Verify Key & Detect Models"):
                with st.spinner("Contacting Google API..."):
                    ok, found_models, msg = AnomalyAIAgent.list_available_models(gemini_api_key)
                    if ok and found_models:
                        st.success(f"Key Verified! Active models ({len(found_models)}): {', '.join(found_models[:5])}")
                    else:
                        st.error(f"Verification issue: {msg}")
        else:
            st.info("💡 Without an API key, the agent will use its built-in rule intelligence engine.")

    with st.expander("📧 Email Alert (SMTP) Settings", expanded=False):
        smtp_preset = st.selectbox(
            "Preset Provider",
            ["Gmail", "Outlook / Office365", "Custom SMTP"]
        )
        
        if smtp_preset == "Gmail":
            default_host = "smtp.gmail.com"
            default_port = 587
        elif smtp_preset == "Outlook / Office365":
            default_host = "smtp.office365.com"
            default_port = 587
        else:
            default_host = get_secret("SMTP_SERVER", "smtp.example.com")
            default_port = int(get_secret("SMTP_PORT", "587"))

        smtp_server = st.text_input("SMTP Server", value=default_host)
        smtp_port = st.number_input("SMTP Port", value=default_port, step=1)
        sender_email = st.text_input("Sender Email", value=get_secret("SENDER_EMAIL", ""))
        sender_password = st.text_input(
            "App Password / Password",
            value=get_secret("SENDER_PASSWORD", ""),
            type="password",
            help="For Gmail, create an App Password in your Google Account security settings."
        )
        default_recipients = st.text_input("Default Recipient(s)", value=get_secret("ALERT_RECIPIENT", ""))

        if st.button("🧪 Test SMTP Connection"):
            if not sender_email or not sender_password:
                st.warning("Please provide sender email and password to test.")
            else:
                with st.spinner("Testing SMTP connection..."):
                    tester = EmailAlerter(
                        smtp_server=smtp_server,
                        smtp_port=smtp_port,
                        sender_email=sender_email,
                        sender_password=sender_password
                    )
                    success, msg = tester.test_connection()
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

    with st.expander("🔬 Detection Parameters", expanded=True):
        algorithm = st.selectbox(
            "Anomaly Detection Algorithm",
            ["Isolation Forest", "Local Outlier Factor (LOF)", "Z-Score (Statistical)", "Ensemble (IF + Z-Score)"],
            help="Isolation Forest is ideal for multi-attribute outliers; LOF captures local density clusters; Z-score catches extreme univariate tails."
        )
        contamination_pct = st.slider(
            "Expected Outlier Ratio (%)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Approximate percentage of records anticipated to be anomalous."
        )
        contamination_rate = contamination_pct / 100.0

    st.markdown("---")
    st.markdown("### 📌 Workflow Overview")
    st.markdown("""
    1. **Upload Dataset** (CSV / Excel)
    2. **Run Anomaly Detection**
    3. **AI Agent Summarization**
    4. **Dispatch Email Alerts**
    5. **Explore Clear Visual Results**
    """)

# Main Screen Header
st.markdown('<div class="main-header">🛡️ Autonomous AI Anomaly Detection Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload any dataset to detect outliers with ML, synthesize insights with Gemini AI, and dispatch automated email alerts.</div>', unsafe_allow_html=True)

# Tabs Layout
tab_data, tab_detect, tab_ai, tab_alert, tab_results = st.tabs([
    "📁 1. Dataset Ingestion",
    "⚙️ 2. Anomaly Detection",
    "🧠 3. AI Agent Summary",
    "📧 4. Email Alerts",
    "📊 5. Visual Dashboard & Clear Results"
])

# ==========================================
# TAB 1: DATASET INGESTION
# ==========================================
with tab_data:
    st.subheader("Step 1: Provide Dataset or File")
    
    col_upload, col_sample = st.columns([3, 2])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload your dataset file (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="Supports CSV and Excel tabular data with numerical attributes."
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state.df_raw = df
                st.session_state.detection_results = None
                st.session_state.ai_analysis = None
                st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df):,} records, {len(df.columns)} columns)")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    with col_sample:
        st.markdown("##### Or load realistic sample dataset:")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("💳 Financial Fraud Sample", use_container_width=True):
                st.session_state.df_raw = generate_financial_dataset(500, anomaly_ratio=0.05)
                st.session_state.detection_results = None
                st.session_state.ai_analysis = None
                st.success("Loaded Financial Transactions sample (500 records)")
        with col_s2:
            if st.button("🖥️ Server Metrics Sample", use_container_width=True):
                st.session_state.df_raw = generate_server_metrics_dataset(500, anomaly_ratio=0.05)
                st.session_state.detection_results = None
                st.session_state.ai_analysis = None
                st.success("Loaded Server Performance Telemetry (500 records)")

    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.markdown("---")
        st.markdown("#### Dataset Profile & Preview")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", f"{len(df):,}")
        m2.metric("Total Features", f"{len(df.columns)}")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        m3.metric("Numeric Features", f"{len(num_cols)}")
        missing_count = int(df.isna().sum().sum())
        m4.metric("Missing Values", f"{missing_count:,}")
        
        st.dataframe(df.head(10), use_container_width=True)
        
        if len(num_cols) == 0:
            st.error("⚠️ No numerical columns found in this dataset. Anomaly detection requires at least one numeric feature.")
        else:
            st.info(f"Detected {len(num_cols)} numeric attributes for anomaly inspection: {', '.join(num_cols)}")
    else:
        st.info("👆 Please upload a file or click one of the sample dataset buttons to proceed.")

# ==========================================
# TAB 2: ANOMALY DETECTION
# ==========================================
with tab_detect:
    st.subheader("Step 2: Run Anomaly Detection")
    
    if st.session_state.df_raw is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        col_ctrl1, col_ctrl2 = st.columns([3, 1])
        with col_ctrl1:
            st.write(f"Configured Engine: **{algorithm}** | Contamination Sensitivity: **{contamination_pct}%**")
        with col_ctrl2:
            run_btn = st.button("⚡ Run Detection Now", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Analyzing dataset, scaling features, and training detector..."):
                try:
                    detector = AnomalyDetector(
                        method=algorithm,
                        contamination=contamination_rate
                    )
                    results = detector.detect(st.session_state.df_raw)
                    st.session_state.detection_results = results
                    st.session_state.ai_analysis = None # Reset AI summary for new run
                    st.success("Anomaly detection completed successfully!")
                except Exception as e:
                    st.error(f"Detection failed: {e}")

        if st.session_state.detection_results is not None:
            res = st.session_state.detection_results
            metrics = res["summary_metrics"]
            
            st.markdown("### Detection Scorecard")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Evaluated", f"{metrics['total_records']:,}")
            c2.metric("Anomalies Found", f"{metrics['anomaly_count']:,}", delta=f"{metrics['anomaly_percentage']}% of dataset", delta_color="inverse")
            c3.metric("Normal Records", f"{metrics['normal_count']:,}")
            c4.metric("Critical High-Risk", f"{metrics['critical_count']:,}", delta="Score > 0.80", delta_color="inverse")

            st.markdown("#### Top Detected Anomalous Records Preview")
            anomaly_df = res["anomaly_df"]
            if not anomaly_df.empty:
                st.dataframe(anomaly_df.head(8), use_container_width=True)
            else:
                st.write("No anomalous records detected with current sensitivity threshold.")

# ==========================================
# TAB 3: AI AGENT SUMMARIZATION
# ==========================================
with tab_ai:
    st.subheader("Step 3: AI Agent Summarization & Root Cause Analysis")
    
    if st.session_state.detection_results is None:
        st.warning("Please run anomaly detection in Step 2 first.")
    else:
        results = st.session_state.detection_results
        metrics = results["summary_metrics"]
        anomaly_samples = results["anomaly_df"].head(10).to_dict(orient="records")

        col_ai1, col_ai2 = st.columns([3, 1])
        with col_ai1:
            st.markdown("Let the AI Agent interpret the detected anomalies, isolate key drivers, and suggest mitigation steps.")
        with col_ai2:
            generate_ai_btn = st.button("🧠 Synthesize Insights", type="primary", use_container_width=True)

        # Trigger generation if clicked or if empty
        if generate_ai_btn or (st.session_state.ai_analysis is None):
            with st.spinner("AI Agent is reasoning over anomaly metrics and formulating briefing..."):
                agent = AnomalyAIAgent(
                    api_key=gemini_api_key if gemini_api_key else None,
                    model_name=gemini_model
                )
                analysis = agent.generate_summary(metrics, anomaly_samples)
                st.session_state.ai_analysis = analysis

        if st.session_state.ai_analysis is not None:
            analysis = st.session_state.ai_analysis
            source_name = analysis.get("source", "AI System")
            
            # Show provider status
            if "Google Gemini" in source_name:
                st.success(f"✨ Intelligence Provider: **{source_name}**")
            else:
                st.info(f"ℹ️ Intelligence Provider: **{source_name}**")

            # Surface any Gemini API issues if present
            if analysis.get("error_details"):
                st.warning(f"⚠️ **Gemini API Notice:** {analysis['error_details']}\n\nFalling back to intelligent rule-based analysis. Please verify your API key or quota if you intended to use Google Gemini.")

            # Executive Summary Card
            with st.container():
                st.markdown("#### 📋 Executive Anomaly Brief")
                st.markdown(analysis.get("executive_summary", ""))

            st.markdown("---")

            col_root, col_recom = st.columns(2)
            with col_root:
                st.markdown("#### 🔍 Root Causes & Primary Drivers")
                st.markdown(analysis.get("root_causes", ""))

            with col_recom:
                st.markdown("#### 💡 Recommended Next Actions")
                st.markdown(analysis.get("recommendations", ""))

            # Pre-drafted Email alert subject & body preview
            with st.expander("📬 AI Pre-Drafted Alert Notification", expanded=False):
                st.text_input("Subject Line", value=analysis.get("email_subject", ""), disabled=True)
                st.text_area("Body Preview", value=analysis.get("email_body", ""), height=180, disabled=True)

# ==========================================
# TAB 4: EMAIL ALERTS
# ==========================================
with tab_alert:
    st.subheader("Step 4: Dispatch Email Alerts")
    
    if st.session_state.detection_results is None:
        st.warning("Please execute anomaly detection in Step 2 first.")
    else:
        results = st.session_state.detection_results
        metrics = results["summary_metrics"]
        anomaly_df = results["anomaly_df"]

        # Ensure AI summary is available
        if st.session_state.ai_analysis is None:
            agent = AnomalyAIAgent(api_key=gemini_api_key if gemini_api_key else None, model_name=gemini_model)
            st.session_state.ai_analysis = agent.generate_summary(metrics, anomaly_df.head(10).to_dict(orient="records"))
            
        analysis = st.session_state.ai_analysis

        col_form, col_preview = st.columns([1, 1])

        with col_form:
            st.markdown("#### Alert Dispatcher")
            recipients_input = st.text_input(
                "Recipient Email(s)",
                value=default_recipients,
                placeholder="ops-team@company.com, manager@company.com",
                help="Separate multiple recipients with commas."
            )
            
            subject_input = st.text_input(
                "Email Subject Line",
                value=analysis.get("email_subject", f"🚨 Anomaly Alert: {metrics['anomaly_count']} Outliers Detected")
            )
            
            attach_csv_option = st.checkbox("Attach full anomalous records CSV", value=True)

            send_alert_btn = st.button("🚀 Send Email Alert Now", type="primary", use_container_width=True)

            if send_alert_btn:
                recipient_list = [r.strip() for r in recipients_input.split(",") if r.strip()]
                if not recipient_list:
                    st.error("Please enter at least one recipient email address.")
                elif not sender_email or not sender_password:
                    st.error("Sender email and password/app password are required. Configure them in the left sidebar.")
                else:
                    with st.spinner("Dispatching HTML alert with metrics and attachment..."):
                        alerter = EmailAlerter(
                            smtp_server=smtp_server,
                            smtp_port=smtp_port,
                            sender_email=sender_email,
                            sender_password=sender_password
                        )
                        success, message = alerter.send_alert(
                            recipient_emails=recipient_list,
                            subject=subject_input,
                            summary_metrics=metrics,
                            ai_analysis=analysis,
                            anomaly_df=anomaly_df,
                            attach_csv=attach_csv_option
                        )
                        if success:
                            st.success(message)
                            st.session_state.email_log.append({
                                "status": "Success",
                                "recipients": ", ".join(recipient_list),
                                "subject": subject_input
                            })
                        else:
                            st.error(message)

        with col_preview:
            st.markdown("#### Live Email Preview")
            alerter_preview = EmailAlerter()
            html_preview = alerter_preview.build_html_report(metrics, analysis, anomaly_df)
            
            # Display rendered HTML in iframe
            st.components.v1.html(html_preview, height=520, scrolling=True)

# ==========================================
# TAB 5: VISUAL DASHBOARD & CLEAR RESULTS
# ==========================================
with tab_results:
    st.subheader("Step 5: Visual Dashboard & Clear Results")
    
    if st.session_state.detection_results is None:
        st.warning("Please run anomaly detection in Step 2 to view interactive visual results.")
    else:
        results = st.session_state.detection_results
        result_df = results["result_df"]
        anomaly_df = results["anomaly_df"]
        metrics = results["summary_metrics"]
        numeric_cols = metrics["numeric_columns"]

        # Metric summary banner
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Evaluated Records", f"{metrics['total_records']:,}")
        m_col2.metric("Detected Anomalies", f"{metrics['anomaly_count']:,}", f"{metrics['anomaly_percentage']}%")
        m_col3.metric("Critical Anomalies", f"{metrics['critical_count']:,}")
        m_col4.metric("Algorithm", f"{metrics['method_used']}")

        st.markdown("---")

        # Visual Chart Row 1: Interactive Scatter Plot & Anomaly Score Distribution
        col_chart1, col_chart2 = st.columns([3, 2])

        with col_chart1:
            st.markdown("##### 📍 Interactive Feature Space & Anomaly Clustering")
            if len(numeric_cols) >= 2:
                c_x = st.selectbox("X-Axis Feature", numeric_cols, index=0, key="x_feat")
                c_y = st.selectbox("Y-Axis Feature", numeric_cols, index=min(1, len(numeric_cols)-1), key="y_feat")
                
                fig_scatter = px.scatter(
                    result_df,
                    x=c_x,
                    y=c_y,
                    color="severity",
                    color_discrete_map={
                        "Normal": "#94a3b8",
                        "Low": "#38bdf8",
                        "Medium": "#f59e0b",
                        "High / Critical": "#ef4444"
                    },
                    hover_data=["anomaly_score", "primary_anomaly_factor"],
                    title=f"Outliers vs Normal Records: {c_x} vs {c_y}",
                    template="plotly_white"
                )
                fig_scatter.update_traces(marker=dict(size=8, opacity=0.85))
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("At least 2 numeric features required for 2D scatter visualization.")

        with col_chart2:
            st.markdown("##### 📈 Anomaly Score Distribution")
            fig_hist = px.histogram(
                result_df,
                x="anomaly_score",
                color="severity",
                nbins=30,
                color_discrete_map={
                    "Normal": "#94a3b8",
                    "Low": "#38bdf8",
                    "Medium": "#f59e0b",
                    "High / Critical": "#ef4444"
                },
                title="Continuous Anomaly Score Spectrum",
                template="plotly_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Time-Series Anomaly Timeline (if any datetime column exists)
        date_cols = [c for c in result_df.columns if "time" in c.lower() or "date" in c.lower()]
        if date_cols and len(numeric_cols) > 0:
            st.markdown("##### ⏱️ Anomaly Timeline Evolution")
            time_col = date_cols[0]
            metric_col = st.selectbox("Select Metric for Time-Series Inspection", numeric_cols, index=0, key="ts_metric")
            
            fig_time = px.scatter(
                result_df.sort_values(by=time_col),
                x=time_col,
                y=metric_col,
                color="severity",
                color_discrete_map={
                    "Normal": "#cbd5e1",
                    "Low": "#38bdf8",
                    "Medium": "#f59e0b",
                    "High / Critical": "#dc2626"
                },
                title=f"{metric_col} Over Time with Highlighted Anomalies",
                template="plotly_white"
            )
            fig_time.update_traces(marker=dict(size=7))
            st.plotly_chart(fig_time, use_container_width=True)

        # Feature Deviation Impact Analysis
        st.markdown("##### ⚖️ Feature Deviation Analysis (Normal Baseline vs Anomaly Cohort)")
        col_impact = metrics.get("column_impact", {})
        if col_impact:
            impact_rows = []
            for col, stats in col_impact.items():
                impact_rows.append({
                    "Feature": col,
                    "Normal Cohort Mean": stats["normal_mean"],
                    "Anomaly Cohort Mean": stats["anomaly_mean"],
                    "Deviation (%)": stats["deviation_pct"]
                })
            impact_df = pd.DataFrame(impact_rows).sort_values(by="Deviation (%)", key=abs, ascending=False)
            st.dataframe(impact_df, use_container_width=True)

        # Filterable Anomalies Table
        st.markdown("##### 📋 Filterable Flagged Anomalies")
        sev_filter = st.multiselect(
            "Filter by Severity",
            ["High / Critical", "Medium", "Low"],
            default=["High / Critical", "Medium", "Low"]
        )
        filtered_anomalies = anomaly_df[anomaly_df["severity"].isin(sev_filter)]
        st.dataframe(filtered_anomalies, use_container_width=True)

        # Download Exports
        st.markdown("##### 💾 Export Results")
        d1, d2, d3 = st.columns(3)
        
        with d1:
            csv_anom = filtered_anomalies.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Flagged Anomalies (CSV)",
                data=csv_anom,
                file_name="detected_anomalies.csv",
                mime="text/csv",
                use_container_width=True
            )

        with d2:
            csv_full = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Enriched Dataset (CSV)",
                data=csv_full,
                file_name="enriched_dataset_with_scores.csv",
                mime="text/csv",
                use_container_width=True
            )

        with d3:
            if st.session_state.ai_analysis:
                summary_text = (
                    f"# AI Anomaly Detection Report\n\n"
                    f"{st.session_state.ai_analysis.get('executive_summary', '')}\n\n"
                    f"## Root Causes\n{st.session_state.ai_analysis.get('root_causes', '')}\n\n"
                    f"## Recommendations\n{st.session_state.ai_analysis.get('recommendations', '')}\n"
                )
                st.download_button(
                    label="📄 Download AI Summary (Markdown)",
                    data=summary_text.encode('utf-8'),
                    file_name="anomaly_summary_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )

st.markdown("---")
st.caption("AI Anomaly Detection Agent • Powered by Scikit-Learn & Google Gemini")
