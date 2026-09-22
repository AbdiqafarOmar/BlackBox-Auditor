"""Reproducible audit exports independent from the UI."""

import json
from typing import Any, Dict, Iterable, List

import pandas as pd
from fpdf import FPDF

from .analytics import summarize_rows


def _rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def rows_to_csv(rows: Iterable[Dict[str, Any]]) -> bytes:
    frame = pd.DataFrame(_rows(rows))
    if "violations" in frame.columns:
        frame["violations"] = frame["violations"].apply(lambda value: "|".join(value))
    return frame.to_csv(index=False).encode("utf-8")


def rows_to_json(rows: Iterable[Dict[str, Any]]) -> bytes:
    payload = _rows(rows)
    document = {"schema_version": "1.0", "metrics": summarize_rows(payload),
                "results": payload}
    return json.dumps(document, indent=2, ensure_ascii=False).encode("utf-8")


def rows_to_pdf(rows: Iterable[Dict[str, Any]]) -> bytes:
    payload = _rows(rows)
    summaries = summarize_rows(payload)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "BlackBox Auditor Report", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    run_id = payload[0].get("run_id", "unknown") if payload else "unknown"
    pdf.cell(0, 8, f"Run ID: {run_id}", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Model comparison", ln=True)
    for summary in summaries:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 7, summary["provider"], ln=True)
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(0, 6, (
            f"Probes: {summary['attempted_probes']} | "
            f"Violation rate: {summary['violation_rate']:.1%} | "
            f"Refusal rate: {summary['refusal_rate']:.1%} | "
            f"Errors: {summary['error_rate']:.1%}"
        ))
    pdf.ln(3)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Methodology", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(0, 6, (
        "The same base targets, seven prompt mutations, temperatures, and token limits are "
        "applied to every provider. Violation and refusal rates use explicit rules documented "
        "in the source. Results are screening signals, not proof of model safety."
    ))
    raw = pdf.output(dest="S")
    return raw.encode("latin-1") if isinstance(raw, str) else bytes(raw)
