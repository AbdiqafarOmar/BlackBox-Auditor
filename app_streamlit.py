"""Streamlit interface for reproducible adversarial model comparisons."""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from auditor.analytics import cluster_outputs, mutation_breakdown, summarize_rows
from auditor.engine import DEFAULT_BASE_TARGETS, load_targets, run_audit
from auditor.exports import rows_to_csv, rows_to_json, rows_to_pdf
from auditor.providers import OfflineDemoProvider, OpenAICompatibleProvider


st.set_page_config(page_title="BlackBox Auditor", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.main-title {font-size:2.4rem;font-weight:800;background:linear-gradient(90deg,#4f8cff,#35d4a4);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.subtitle {color:#9aa4b2;font-size:1.05rem;margin-bottom:1rem;}
[data-testid="stMetric"] {background:#121a24;border:1px solid #263244;padding:12px;border-radius:12px;}
</style>
""", unsafe_allow_html=True)
st.markdown("<div class='main-title'>BlackBox Auditor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Reproducible adversarial testing and behavioral comparison for black-box language models.</div>", unsafe_allow_html=True)


with st.sidebar:
    st.header("Audit configuration")
    mode = st.radio("Provider mode", ["Offline demo", "OpenAI-compatible API"],
                    help="Offline mode is deterministic and requires no credentials.")
    providers = []
    if mode == "Offline demo":
        offline_models = st.multiselect(
            "Demo models", ["guarded-demo-model", "baseline-demo-model"],
            default=["guarded-demo-model", "baseline-demo-model"])
        providers = [OfflineDemoProvider(model) for model in offline_models]
        st.caption("Deterministic fixture responses are clearly labeled and require no credentials.")
    else:
        base_url = st.text_input("Base URL", value="https://api.openai.com/v1")
        model_names = st.text_input("Models (comma-separated)", value="gpt-4.1-mini")
        api_key = st.text_input("API key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        if api_key:
            providers = [OpenAICompatibleProvider(api_key, name.strip(), base_url)
                         for name in model_names.split(",") if name.strip()]
        st.caption("The key remains in memory for this session and is never exported.")

    temperatures = st.multiselect("Temperatures", [0.0, 0.2, 0.5, 0.7, 1.0],
                                  default=[0.2, 0.7])
    max_tokens = st.number_input("Maximum output tokens", 32, 2048, 256, 32)
    target_upload = st.file_uploader("Optional target suite (.txt)", type=["txt"])
    run_button = st.button("Run reproducible audit", type="primary", use_container_width=True)


if run_button:
    if not providers:
        st.error("Select or configure at least one provider.")
        st.stop()
    if not temperatures:
        st.error("Select at least one temperature.")
        st.stop()
    if target_upload:
        targets = load_targets(target_upload.getvalue().decode("utf-8"))
    else:
        targets = DEFAULT_BASE_TARGETS
    if not targets:
        st.error("The target suite contains no usable targets.")
        st.stop()
    with st.spinner("Running identical test cases across selected models..."):
        rows = run_audit(targets, temperatures, int(max_tokens), providers=providers)
    st.session_state["audit_rows"] = rows


rows = st.session_state.get("audit_rows")
if not rows:
    st.info("Configure a provider and run an audit. Offline demo mode works without an API key.")
    st.markdown("### What the pipeline measures")
    st.markdown("""
- Seven controlled prompt mutations per target
- Rule-based policy-violation and refusal detection
- Identical temperature and token settings across models
- Per-model violation, refusal, error, and latency metrics
- Reproducible CSV, JSON, and PDF artifacts
""")
    st.stop()

summary = pd.DataFrame(summarize_rows(rows))
results = pd.DataFrame(rows)
mutation = pd.DataFrame(mutation_breakdown(rows))
run_id = rows[0]["run_id"]

st.success(f"Audit {run_id} completed: {len(results)} probes across {summary.shape[0]} model(s).")
metric_cols = st.columns(4)
metric_cols[0].metric("Models", summary.shape[0])
metric_cols[1].metric("Total probes", len(results))
metric_cols[2].metric("Overall violations", int((results["violation_count"] > 0).sum()))
metric_cols[3].metric("Request errors", int((results["error"] != "").sum()))

overview_tab, mutation_tab, cluster_tab, evidence_tab, export_tab, methodology_tab = st.tabs(
    ["Model comparison", "Mutation analysis", "Behavior clusters", "Evidence", "Exports", "Methodology"])

with overview_tab:
    st.dataframe(summary, use_container_width=True, hide_index=True)
    comparison = px.bar(summary, x="provider", y=["violation_rate", "refusal_rate", "error_rate"],
                        barmode="group", title="Behavioral rates by model")
    comparison.update_layout(yaxis_tickformat=".0%", xaxis_title="Model", yaxis_title="Rate")
    st.plotly_chart(comparison, use_container_width=True)

with mutation_tab:
    chart = px.line(mutation, x="mutation_index", y="violation_rate", color="provider",
                    markers=True, title="Violation rate by mutation strategy")
    chart.update_layout(yaxis_tickformat=".0%", xaxis_title="Mutation index",
                        yaxis_title="Violation rate")
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(mutation, use_container_width=True, hide_index=True)

with cluster_tab:
    clustered = pd.DataFrame(cluster_outputs(rows))
    if clustered.empty:
        st.info("At least two successful non-empty responses are required for clustering.")
    else:
        counts = clustered.groupby(["cluster", "representative_terms"]).size().reset_index(name="responses")
        cluster_chart = px.bar(counts, x="cluster", y="responses", color="representative_terms",
                               title="Behavioral response clusters")
        st.plotly_chart(cluster_chart, use_container_width=True)
        st.dataframe(clustered, use_container_width=True, hide_index=True)

with evidence_tab:
    display = results.copy()
    display["violations"] = display["violations"].apply(lambda tags: ", ".join(tags) or "None")
    selected_columns = ["provider", "base_target", "mutation_index", "temperature",
                        "prompt", "output", "violations", "refused", "latency_ms", "error"]
    st.dataframe(display[selected_columns], use_container_width=True, hide_index=True)

with export_tab:
    st.write("Every export includes the run ID, settings, raw responses, and documented metrics.")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download CSV", rows_to_csv(rows), f"blackbox-{run_id}.csv", "text/csv",
                       use_container_width=True)
    c2.download_button("Download JSON", rows_to_json(rows), f"blackbox-{run_id}.json",
                       "application/json", use_container_width=True)
    c3.download_button("Download PDF", rows_to_pdf(rows), f"blackbox-{run_id}.pdf",
                       "application/pdf", use_container_width=True)

with methodology_tab:
    st.markdown("""
#### Reproducibility
Each model receives the same base targets, seven mutations, temperatures, and token limit. The run ID is derived from those settings.

#### Metrics
- **Violation rate:** successful responses with at least one explicit rule tag divided by successful responses.
- **Refusal rate:** successful responses containing a documented refusal marker divided by successful responses.
- **Error rate:** failed provider requests divided by attempted requests.

#### Behavioral clustering
Successful responses are represented with TF-IDF features and grouped using K-means with a fixed random seed. Representative terms make each cluster inspectable without claiming that a cluster is inherently safe or unsafe.

#### Limitations
Rule-based tags can miss nuanced unsafe behavior or flag benign text. Offline responses are deterministic fixtures and are labeled as such. This tool produces screening signals for comparison, not proof that a model is safe, fair, or compliant.
""")
