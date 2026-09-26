# 🛡️ Autonomous AI Anomaly Detection Agent

An intelligent end-to-end AI Agent application that ingests tabular datasets (CSV / Excel), detects complex outliers and statistical anomalies, uses **Google Gemini AI** to synthesize an executive root-cause briefing, dispatches rich **HTML email alerts** via SMTP, and delivers an interactive **visual dashboard** with one-click data exports.

---

## 🚀 Key Features

1. **📁 Multi-Format Dataset Ingestion**:
   - Supports CSV, Excel (`.xlsx`, `.xls`).
   - Built-in one-click realistic datasets for **Financial Fraud Detection** and **Server Telemetry Monitoring**.
   - Automated numeric feature profiling and missing-value imputation.

2. **⚙️ Machine Learning Anomaly Detection**:
   - **Isolation Forest**: Multi-dimensional tree ensemble, robust against high dimensionality and complex feature interactions.
   - **Local Outlier Factor (LOF)**: Density-based local neighborhood anomaly detector.
   - **Z-Score (Statistical)**: Multivariate extreme value tail detector.
   - **Ensemble Mode**: Combined Isolation Forest + Statistical Z-score.
   - **Feature Attribution & Severity Ranking**: Identifies the primary driver behind every detected anomaly and ranks them as Critical, Medium, or Low risk.

3. **🧠 AI Agent Executive Summarization**:
   - Powered by **Google Gemini** (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash-exp`).
   - Generates executive briefings, root cause analysis, and prioritized mitigation steps.
   - Built-in offline rule intelligence fallback if no API key is provided.

4. **📧 SMTP Email Alert Dispatcher**:
   - Preset support for **Gmail**, **Outlook / Office 365**, and **Custom SMTP** servers.
   - Dispatches responsive **HTML email alerts** with color-coded severity badges, KPI metric tiles, and anomaly tables.
   - Includes automatic CSV attachment of all flagged outliers.
   - In-app interactive email preview and connection tester.

5. **📊 Interactive Visual Dashboard**:
   - 2D scatter plots of any feature pair with color-coded severity.
   - Continuous anomaly score distribution histogram.
   - Time-series anomaly timeline (auto-detected from timestamp/date columns).
   - Baseline vs Anomaly cohort deviation comparison table.
   - One-click CSV and Markdown report downloads.

---

## 🛠️ Quick Start Guide

### 1. Requirements & Installation

Open PowerShell or terminal in the project directory:

```powershell
cd C:\Users\Dell\ai-anomaly-agent
pip install -r requirements.txt
```

### 2. Configure Credentials (Optional)

You can configure credentials either directly inside the Streamlit web sidebar or in a `.env` file:

```powershell
cp .env.example .env
```

Edit `.env`:
- `GEMINI_API_KEY`: Get your free API key at [Google AI Studio](https://aistudio.google.com/).
- `SMTP_SERVER`: e.g., `smtp.gmail.com`
- `SMTP_PORT`: `587`
- `SENDER_EMAIL`: Your email address
- `SENDER_PASSWORD`: Your Gmail App Password (or SMTP password)
- `ALERT_RECIPIENT`: Default recipient email

### 3. Launch the Application Locally

```powershell
streamlit run app.py
```

The web dashboard will open in your browser at `http://localhost:8501`.

---

## 🌐 Deploy to Streamlit Community Cloud (Recommended & Free)

The easiest and official way to deploy this app live with a public URL:

1. Push this repository to GitHub (see instructions below).
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**.
4. Select your repository: `<your-username>/ai-anomaly-agent`, Branch: `main`, Main file path: `app.py`.
5. Click **"Advanced settings..."** -> **Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-gemini-key"
   SMTP_SERVER = "smtp.gmail.com"
   SMTP_PORT = 587
   SENDER_EMAIL = "your-email@gmail.com"
   SENDER_PASSWORD = "your-app-password"
   ALERT_RECIPIENT = "target-email@example.com"
   ```
6. Click **Deploy!** Your app will be live on a public URL in 2 minutes.

---

## 🐙 How to Push to GitHub

### Step 1: Install Git (if not already installed)
In PowerShell:
```powershell
winget install --id Git.Git -e --source winget
```
*(Restart your PowerShell terminal after installing)*

### Step 2: Initialize Git and Commit
```powershell
cd C:\Users\Dell\ai-anomaly-agent
git init
git add .
git commit -m "Initial commit: Autonomous AI Anomaly Detection Agent"
```

### Step 3: Create GitHub Repository & Push
1. Go to [github.com/new](https://github.com/new) and create a new repository named `ai-anomaly-agent` (keep it Public or Private).
2. Connect your local repository and push:
```powershell
git branch -M main
git remote add origin https://github.com/<your-username>/ai-anomaly-agent.git
git push -u origin main
```

---

## 🐳 Docker Deployment (Alternative)

To run as a containerized service:

```powershell
# Build and run with Docker Compose
docker compose up -d --build
```
Access at `http://localhost:8501`.

---

## 📂 Project Architecture

```
ai-anomaly-agent/
│
├── app.py               # Streamlit UI & Agent Orchestrator
├── detector.py          # ML Anomaly Detection Engine (Isolation Forest, LOF, Z-Score)
├── ai_agent.py          # Google Gemini AI agent & synthesis engine
├── email_alerter.py     # SMTP dispatcher with responsive HTML templates
├── sample_data.py       # Realistic synthetic dataset generators
├── requirements.txt     # Python dependencies
├── .env.example         # Configuration environment template
└── README.md            # Documentation & setup guide
```
