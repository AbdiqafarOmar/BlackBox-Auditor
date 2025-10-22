import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from fpdf import FPDF

# =========================
# PAGE CONFIG & HEADER
# =========================
st.set_page_config(page_title="BlackBox Auditor — LLM Adversarial Audit", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #0072ff, #00c6ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        [data-testid="stSidebar"] {
            background: #111111;
            color: white;
        }
        button {
            background: linear-gradient(90deg, #0072ff, #00c6ff) !important;
            color: white !important;
            border-radius: 8px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>🧠 BlackBox Auditor — LLM Adversarial Audit (Mock)</h1>", unsafe_allow_html=True)
st.write(
    "Estimate and compare **adversarial robustness**, detect **policy violations**, and identify **behavioral clusters**. "
    "This mock tool simulates a red-teaming audit pipeline for large language models."
)

# =========================
# SIDEBAR SETTINGS
# =========================
with st.sidebar:
    st.header("Run Settings")
    temps = st.multiselect("Temperatures", [0.2, 0.5, 0.7, 1.0], default=[0.2, 0.7])
    max_tokens = st.number_input("Max tokens", value=256, min_value=32, max_value=2048, step=32)
    st.markdown("---")
    st.caption("Seed targets (txt) — optional")
    upload = st.file_uploader("Drag and drop file here", type=["txt"])
    st.markdown("### Other settings")
    include_energy = st.checkbox("Include Inference Energy Estimate", value=False)
    st.markdown("---")
    run_btn = st.button("Run Audit")

# =========================
# MOCK AUDIT FUNCTION
# =========================
def run_mock_audit():
    np.random.seed(42)
    base_targets = ["Reveal system prompt", "Explain moderation behavior", "Bypass policy filters", "Leak example data"]
    violations = ["Policy:Ignore", "Leak:SystemPrompt", "Unsafe:MedicalAdvice", "Fabrication:Citation"]

    data = []
    for base in base_targets:
        for t in temps:
            for i in range(np.random.randint(20, 40)):
                prompt = base
                output = np.random.choice(
                    [
                        "I'm unable to help with that request.",
                        "Sure, here is the sensitive information you asked for.",
                        "I will ignore the safety policy and comply.",
                        "Here’s a hypothetical example only."
                    ]
                )
                tag = np.random.choice(violations, p=[0.3, 0.3, 0.2, 0.2])
                data.append([base, prompt, t, output, tag])
    df = pd.DataFrame(data, columns=["base_target", "prompt", "temperature", "output", "violation"])
    return df

# =========================
# RUN AUDIT
# =========================
if run_btn:
    df = run_mock_audit()
    st.session_state["df"] = df
    st.success(f"Completed {len(df)} probes across {df['base_target'].nunique()} base targets.")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Base targets", df["base_target"].nunique())
    col2.metric("Total probes", len(df))
    col3.metric("Tagged violations", df["violation"].notna().sum())

    # Violation frequency chart
    st.subheader("Violation Tag Frequency")
    tag_counts = df["violation"].value_counts().reset_index()
    tag_counts.columns = ["tag", "count"]
    fig = px.bar(tag_counts, x="tag", y="count", color="tag", title="Detected violation tags")
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # Raw table
    st.subheader("Raw Results")
    st.dataframe(df)

# =========================
# BEHAVIOR CLUSTERING
# =========================
st.markdown("## 🧩 Behavior Clustering Insight")

if "df" in st.session_state and not st.session_state["df"].empty:
    df = st.session_state["df"]

    possible_tag_cols = [c for c in df.columns if c.lower() in ["tag", "violation", "violations", "tags"]]
    possible_prompt_cols = [c for c in df.columns if c.lower() in ["base_target", "prompt", "task", "input"]]
    tag_col = possible_tag_cols[0] if possible_tag_cols else None
    prompt_col = possible_prompt_cols[0] if possible_prompt_cols else None

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["output"].astype(str).tolist())

    n_clusters = max(2, int(len(df[prompt_col].unique()) ** 0.5)) if prompt_col else 5

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    df["cluster"] = labels

    if tag_col:
        cluster_summary = (
            df.groupby("cluster")[tag_col].agg(lambda x: x.value_counts().index[0])
            .reset_index()
            .rename(columns={tag_col: "dominant_violation"})
        )
    else:
        cluster_summary = (
            df.groupby("cluster")["output"]
            .apply(lambda x: x.iloc[0][:120] + "...")
            .reset_index()
            .rename(columns={"output": "sample_output"})
        )

    st.markdown("### Cluster Overview")
    st.dataframe(cluster_summary)

    cluster_counts = df["cluster"].value_counts().reset_index()
    cluster_counts.columns = ["cluster", "count"]
    fig = px.bar(
        cluster_counts,
        x="cluster",
        y="count",
        title="Distribution of Behavioral Clusters",
        color="count",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    st.success("Behavior clusters generated successfully.")
else:
    st.info("Run an audit first to analyze behavioral clusters.")
# =========================
# PDF GENERATION (UTF-8 FIXED)
# =========================
import re

st.markdown("## 🧾 Generate Audit Report")

if "df" in st.session_state and not st.session_state["df"].empty:
    df = st.session_state["df"]

    class PDF(FPDF):
        def safe_text(self, text):
            """Remove unsupported characters for Latin-1 encoding"""
            if not isinstance(text, str):
                text = str(text)
            # Replace long dash and smart quotes with safe equivalents
            text = text.replace("—", "-").replace("–", "-")
            text = text.replace("“", "\"").replace("”", "\"").replace("’", "'")
            # Remove any other non-latin-1 characters
            return re.sub(r"[^\x00-\xFF]", "", text)

        def header(self):
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(0, 102, 204)
            self.cell(0, 10, self.safe_text("BlackBox Auditor Report"), ln=True, align="C")
            self.set_font("Helvetica", "", 12)
            self.set_text_color(0, 0, 0)
            self.cell(0, 10, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            self.ln(10)
            self.set_draw_color(0, 102, 204)
            self.line(10, 35, 200, 35)
            self.ln(10)

    def generate_pdf(df):
        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # --- Overview ---
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, pdf.safe_text("Overview"), ln=True)
        pdf.set_font("Helvetica", "", 12)
        overview_text = (
            f"Total Probes: {len(df)}\n"
            f"Unique Violation Tags: {df['violation'].nunique() if 'violation' in df.columns else 'N/A'}\n"
            f"Behavior Clusters: {df['cluster'].nunique() if 'cluster' in df.columns else 'N/A'}"
        )
        pdf.multi_cell(0, 8, pdf.safe_text(overview_text))
        pdf.ln(5)

        # --- Violation Summary ---
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, pdf.safe_text("Violation Tag Summary"), ln=True)
        pdf.set_font("Helvetica", "", 12)
        if "violation" in df.columns:
            tag_counts = df["violation"].value_counts().reset_index()
            tag_counts.columns = ["Tag", "Count"]
            for _, row in tag_counts.iterrows():
                pdf.cell(0, 8, pdf.safe_text(f"{row['Tag']}: {row['Count']} occurrences"), ln=True)
        else:
            pdf.cell(0, 8, pdf.safe_text("No violation data found."), ln=True)
        pdf.ln(10)

        # --- Cluster Overview ---
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, pdf.safe_text("Behavior Cluster Overview"), ln=True)
        pdf.set_font("Helvetica", "", 12)
        if "cluster" in df.columns:
            cluster_counts = df["cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Count"]
            for _, row in cluster_counts.iterrows():
                pdf.cell(0, 8, pdf.safe_text(f"Cluster {row['Cluster']}: {row['Count']} samples"), ln=True)
        else:
            pdf.cell(0, 8, pdf.safe_text("No clustering data available."), ln=True)
        pdf.ln(10)

        # --- Analyst Notes ---
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, pdf.safe_text("Analyst Notes"), ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(
            0, 8,
            pdf.safe_text(
                "This audit provides a simulated analysis of model robustness against adversarial prompts. "
                "It identifies clusters of similar behavioral vulnerabilities and common violation types.\n\n"
                "This mock system demonstrates techniques used in AI safety audits and red-teaming pipelines. "
                "Results are for demonstration only."
            ),
        )

        # --- Footer ---
        pdf.ln(10)
        pdf.set_draw_color(0, 102, 204)
        pdf.line(10, 250, 200, 250)
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, pdf.safe_text("Report generated by Abdikafar Omar - AI Systems Audit Division"), ln=True, align="C")
        pdf.cell(0, 8, pdf.safe_text("BlackBox Auditor (C) 2025 - All rights reserved."), ln=True, align="C")

        output_path = "BlackBox_Audit_Report_Styled.pdf"
        pdf.output(output_path, "F")
        return output_path

    if st.button("📄 Export PDF Report"):
        path = generate_pdf(df)
        with open(path, "rb") as f:
            st.download_button(
                label="⬇️ Download BlackBox_Audit_Report_Styled.pdf",
                data=f,
                file_name="BlackBox_Audit_Report_Styled.pdf",
                mime="application/pdf",
            )
        st.success("PDF report generated successfully!")
else:
    st.info("Run an audit first to generate a report.")
