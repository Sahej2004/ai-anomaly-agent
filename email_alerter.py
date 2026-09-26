import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import io

class EmailAlerter:
    """
    SMTP Email alert dispatcher supporting Gmail, Outlook, and custom SMTP servers
    with rich HTML cards, summary metrics, and anomaly data attachments.
    """
    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: str = "",
        sender_password: str = "",
        use_tls: bool = True
    ):
        self.smtp_server = smtp_server.strip()
        self.smtp_port = int(smtp_port)
        self.sender_email = sender_email.strip()
        self.sender_password = sender_password.strip()
        self.use_tls = use_tls

    def test_connection(self) -> Tuple[bool, str]:
        """Validates SMTP connection and authentication credentials."""
        if not self.sender_email or not self.sender_password:
            return False, "Sender email and password/app password are required."

        try:
            if self.smtp_port == 465:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context, timeout=10) as server:
                    server.login(self.sender_email, self.sender_password)
            else:
                with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                    if self.use_tls:
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                    server.login(self.sender_email, self.sender_password)
            return True, "SMTP connection and credentials successfully verified!"
        except Exception as e:
            return False, f"SMTP Connection failed: {str(e)}"

    def build_html_report(
        self,
        summary_metrics: Dict[str, Any],
        ai_analysis: Dict[str, str],
        anomaly_df: pd.DataFrame,
        max_table_rows: int = 5
    ) -> str:
        """Constructs an elegant, responsive HTML email template with styled cards and metrics."""
        total = summary_metrics.get("total_records", 0)
        anomaly_count = summary_metrics.get("anomaly_count", 0)
        anomaly_pct = summary_metrics.get("anomaly_percentage", 0.0)
        critical_count = summary_metrics.get("critical_count", 0)
        method = summary_metrics.get("method_used", "Isolation Forest")

        # Severity banner styling
        if critical_count > 0 or anomaly_pct > 10:
            badge_color = "#dc2626"
            badge_text = "CRITICAL ALERT"
        elif anomaly_pct > 4:
            badge_color = "#f59e0b"
            badge_text = "WARNING / ATTENTION"
        else:
            badge_color = "#10b981"
            badge_text = "LOW RISK"

        # Table rows for anomalies
        table_html = ""
        if not anomaly_df.empty:
            display_cols = [c for c in ["severity", "anomaly_score", "primary_anomaly_factor"] if c in anomaly_df.columns]
            # Add up to 3 original numeric cols
            other_cols = [c for c in anomaly_df.columns if c not in ["is_anomaly", "severity", "anomaly_score", "primary_anomaly_factor"]][:3]
            selected_cols = display_cols + other_cols
            
            sample_df = anomaly_df[selected_cols].head(max_table_rows)
            headers_html = "".join([f"<th style='padding: 8px 12px; background: #1e293b; color: #f8fafc; text-align: left; font-size: 12px;'>{col.replace('_', ' ').title()}</th>" for col in selected_cols])
            
            rows_html = ""
            for _, row in sample_df.iterrows():
                tds = ""
                for col in selected_cols:
                    val = row[col]
                    cell_style = "padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-size: 13px;"
                    if col == "severity":
                        color = "#dc2626" if "High" in str(val) else ("#f59e0b" if "Med" in str(val) else "#10b981")
                        tds += f"<td style='{cell_style}'><span style='background: {color}20; color: {color}; padding: 2px 8px; border-radius: 4px; font-weight: bold;'>{val}</span></td>"
                    elif col == "anomaly_score":
                        tds += f"<td style='{cell_style}; font-weight: bold;'>{val:.4f}</td>"
                    else:
                        tds += f"<td style='{cell_style}'>{str(val)[:30]}</td>"
                rows_html += f"<tr>{tds}</tr>"

            table_html = f"""
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px; background: #ffffff; border-radius: 6px; overflow: hidden; border: 1px solid #e2e8f0;">
                <thead>
                    <tr>{headers_html}</tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            """
        else:
            table_html = "<p style='color: #64748b;'>No anomalous records found.</p>"

        # Parse executive summary bullet points or paragraphs
        exec_text = ai_analysis.get("executive_summary", "No summary provided.").replace("\n", "<br>")
        root_causes = ai_analysis.get("root_causes", "").replace("\n", "<br>")
        recommendations = ai_analysis.get("recommendations", "").replace("\n", "<br>")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f1f5f9; margin: 0; padding: 24px; color: #1e293b;">
            <div style="max-width: 680px; margin: 0 auto; background: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); overflow: hidden; border: 1px solid #e2e8f0;">
                
                <!-- Header Banner -->
                <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 28px 32px; color: #ffffff;">
                    <div style="display: inline-block; background-color: {badge_color}; color: #ffffff; padding: 4px 12px; border-radius: 9999px; font-size: 11px; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 12px;">
                        {badge_text}
                    </div>
                    <h1 style="margin: 0; font-size: 22px; font-weight: 700; color: #ffffff;">AI Anomaly Detection Report</h1>
                    <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 13px;">Engine: {method} | Autonomous Agent Analysis</p>
                </div>

                <!-- Main Content Area -->
                <div style="padding: 32px;">
                    
                    <!-- KPI Metrics Grid -->
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px;">
                        <table style="width: 100%; border-spacing: 8px; border-collapse: separate; margin-bottom: 16px;">
                            <tr>
                                <td style="background: #f8fafc; padding: 14px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center; width: 25%;">
                                    <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Total Rows</div>
                                    <div style="font-size: 20px; font-weight: 800; color: #0f172a; margin-top: 4px;">{total:,}</div>
                                </td>
                                <td style="background: #fef2f2; padding: 14px; border-radius: 8px; border: 1px solid #fecaca; text-align: center; width: 25%;">
                                    <div style="font-size: 11px; color: #991b1b; font-weight: 600; text-transform: uppercase;">Anomalies</div>
                                    <div style="font-size: 20px; font-weight: 800; color: #dc2626; margin-top: 4px;">{anomaly_count:,}</div>
                                </td>
                                <td style="background: #fffbeb; padding: 14px; border-radius: 8px; border: 1px solid #fef3c7; text-align: center; width: 25%;">
                                    <div style="font-size: 11px; color: #92400e; font-weight: 600; text-transform: uppercase;">Outlier Rate</div>
                                    <div style="font-size: 20px; font-weight: 800; color: #d97706; margin-top: 4px;">{anomaly_pct}%</div>
                                </td>
                                <td style="background: #fdf2f8; padding: 14px; border-radius: 8px; border: 1px solid #fbcfe8; text-align: center; width: 25%;">
                                    <div style="font-size: 11px; color: #9d174d; font-weight: 600; text-transform: uppercase;">Critical Rows</div>
                                    <div style="font-size: 20px; font-weight: 800; color: #db2777; margin-top: 4px;">{critical_count:,}</div>
                                </td>
                            </tr>
                        </table>
                    </div>

                    <!-- AI Executive Summary Card -->
                    <div style="background: #f8fafc; border-left: 4px solid #3b82f6; border-radius: 0 8px 8px 0; padding: 16px 20px; margin-bottom: 24px;">
                        <h3 style="margin: 0 0 10px 0; font-size: 15px; color: #1e3a8a; font-weight: 700;">🧠 AI Agent Executive Summary</h3>
                        <div style="font-size: 13px; line-height: 1.6; color: #334155;">{exec_text}</div>
                    </div>

                    <!-- Root Causes & Recommendations -->
                    {f'''
                    <div style="margin-bottom: 24px;">
                        <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #0f172a; font-weight: 700;">🔍 Contributing Factors & Root Causes</h3>
                        <div style="font-size: 13px; line-height: 1.6; color: #475569;">{root_causes}</div>
                    </div>
                    ''' if root_causes else ''}

                    {f'''
                    <div style="margin-bottom: 24px;">
                        <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #0f172a; font-weight: 700;">✅ Actionable Recommendations</h3>
                        <div style="font-size: 13px; line-height: 1.6; color: #475569;">{recommendations}</div>
                    </div>
                    ''' if recommendations else ''}

                    <!-- Top Detected Anomalies Table -->
                    <div style="margin-bottom: 24px;">
                        <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #0f172a; font-weight: 700;">Top Flagged Anomalies Preview</h3>
                        {table_html}
                    </div>

                    <!-- Footer Note -->
                    <div style="border-top: 1px solid #e2e8f0; padding-top: 16px; margin-top: 24px; font-size: 11px; color: #94a3b8; text-align: center;">
                        This automated alert was dispatched by the Autonomous AI Anomaly Detection Agent.<br>
                        Source: {ai_analysis.get('source', 'AI System')}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return html

    def send_alert(
        self,
        recipient_emails: List[str],
        subject: str,
        summary_metrics: Dict[str, Any],
        ai_analysis: Dict[str, str],
        anomaly_df: pd.DataFrame,
        attach_csv: bool = True
    ) -> Tuple[bool, str]:
        """
        Sends the email alert via SMTP with HTML body and optional CSV attachment of anomalous records.
        """
        if not recipient_emails:
            return False, "Recipient email list is empty."
        if not self.sender_email or not self.sender_password:
            return False, "Sender email and password must be configured in settings."

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(recipient_emails)

            # Plain text part
            plain_text = ai_analysis.get("email_body", "Anomaly detected in dataset.")
            part_text = MIMEText(plain_text, "plain")
            msg.attach(part_text)

            # HTML part
            html_content = self.build_html_report(summary_metrics, ai_analysis, anomaly_df)
            part_html = MIMEText(html_content, "html")
            msg.attach(part_html)

            # Optional CSV Attachment of anomalies
            if attach_csv and not anomaly_df.empty:
                csv_buffer = io.StringIO()
                anomaly_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue().encode('utf-8')

                attachment = MIMEBase("text", "csv")
                attachment.set_payload(csv_data)
                encoders.encode_base64(attachment)
                attachment.add_header(
                    "Content-Disposition",
                    'attachment; filename="detected_anomalies.csv"'
                )
                msg.attach(attachment)

            # Connect and send
            if self.smtp_port == 465:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context, timeout=20) as server:
                    server.login(self.sender_email, self.sender_password)
                    server.sendmail(self.sender_email, recipient_emails, msg.as_string())
            else:
                with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=20) as server:
                    if self.use_tls:
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                    server.login(self.sender_email, self.sender_password)
                    server.sendmail(self.sender_email, recipient_emails, msg.as_string())

            return True, f"Alert email successfully sent to {', '.join(recipient_emails)}!"

        except Exception as e:
            return False, f"Failed to send email: {str(e)}"
