"""Reproducible audit exports independent from the UI."""

import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

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
    document = {
        "schema_version": "1.1",
        "run_id": payload[0].get("run_id", "unknown") if payload else "unknown",
        "created_at": payload[0].get("created_at") if payload else None,
        "settings": _settings(payload),
        "metrics": summarize_rows(payload),
        "results": payload,
    }
    return json.dumps(document, indent=2, ensure_ascii=False).encode("utf-8")


def _settings(payload: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recover the shared audit configuration from exported result rows."""
    return {
        "providers": sorted({row.get("provider", "unknown") for row in payload}),
        "base_targets": list(dict.fromkeys(row.get("base_target", "") for row in payload)),
        "temperatures": sorted({float(row.get("temperature", 0.0)) for row in payload}),
        "max_tokens": sorted({int(row.get("max_tokens", 0)) for row in payload}),
        "mutation_count_per_target": len({row.get("mutation_index", 0) for row in payload}),
    }


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def rows_to_pdf(rows: Iterable[Dict[str, Any]]) -> bytes:
    payload = _rows(rows)
    summaries = summarize_rows(payload)
    settings = _settings(payload)
    buffer = BytesIO()
    page_width, page_height = A4
    pdf = canvas.Canvas(buffer, pagesize=A4, pageCompression=1)
    navy = colors.HexColor("#111827")
    slate = colors.HexColor("#475569")
    muted = colors.HexColor("#64748B")
    pale = colors.HexColor("#F1F5F9")
    line = colors.HexColor("#CBD5E1")
    accent = colors.HexColor("#2563EB")
    font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
    pdfmetrics.registerFont(TTFont("DejaVu", str(font_dir / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(font_dir / "VeraBd.ttf")))

    pdf.setFillColor(navy)
    pdf.rect(0, page_height - 39 * mm, page_width, 39 * mm, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("DejaVu-Bold", 21)
    pdf.drawString(15 * mm, page_height - 18 * mm, "BlackBox Auditor")
    pdf.setFillColor(colors.HexColor("#CBD5E1"))
    pdf.setFont("DejaVu", 10)
    pdf.drawString(15 * mm, page_height - 26 * mm,
                   "Reproducible adversarial evaluation report")

    run_id = payload[0].get("run_id", "unknown") if payload else "unknown"
    created_at = payload[0].get("created_at", "unknown") if payload else "unknown"
    y = page_height - 49 * mm
    pdf.setFillColor(muted)
    pdf.setFont("DejaVu", 7.5)
    pdf.drawString(15 * mm, y, f"RUN ID  {run_id}")
    pdf.drawRightString(page_width - 15 * mm, y, f"CREATED  {created_at}")

    total = len(payload)
    violations = sum(row.get("violation_count", 0) > 0 for row in payload)
    errors = sum(bool(row.get("error")) for row in payload)
    y -= 9 * mm
    cards = [("MODELS", str(len(summaries))), ("PROBES", str(total)),
             ("VIOLATIONS", str(violations)), ("ERRORS", str(errors))]
    for index, (label, value) in enumerate(cards):
        x = 15 * mm + index * 46 * mm
        pdf.setFillColor(pale)
        pdf.roundRect(x, y - 20 * mm, 42 * mm, 20 * mm, 2 * mm, fill=1, stroke=0)
        pdf.setFillColor(muted)
        pdf.setFont("DejaVu-Bold", 7)
        pdf.drawString(x + 4 * mm, y - 6 * mm, label)
        pdf.setFillColor(navy)
        pdf.setFont("DejaVu-Bold", 15)
        pdf.drawString(x + 4 * mm, y - 15 * mm, value)

    y -= 31 * mm
    pdf.setFillColor(navy)
    pdf.setFont("DejaVu-Bold", 13)
    pdf.drawString(15 * mm, y, "Model comparison")
    y -= 8 * mm
    widths = [66, 25, 32, 32, 25]
    headers = ["Provider", "Probes", "Violation", "Refusal", "Errors"]
    x = 15 * mm
    pdf.setFillColor(colors.HexColor("#1E293B"))
    pdf.rect(x, y - 8 * mm, sum(widths) * mm, 8 * mm, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("DejaVu-Bold", 8)
    cursor = x
    for width, header in zip(widths, headers):
        pdf.drawCentredString(cursor + width * mm / 2, y - 5.3 * mm, header)
        cursor += width * mm
    y -= 8 * mm
    pdf.setFont("DejaVu", 8)
    for index, summary in enumerate(summaries):
        pdf.setFillColor(colors.HexColor("#F8FAFC") if index % 2 == 0 else colors.white)
        pdf.rect(x, y - 8 * mm, sum(widths) * mm, 8 * mm, fill=1, stroke=0)
        pdf.setStrokeColor(line)
        pdf.line(x, y - 8 * mm, x + sum(widths) * mm, y - 8 * mm)
        pdf.setFillColor(navy)
        values = [summary["provider"], str(summary["attempted_probes"]),
                  _percent(summary["violation_rate"]), _percent(summary["refusal_rate"]),
                  _percent(summary["error_rate"])]
        cursor = x
        for col, (width, value) in enumerate(zip(widths, values)):
            if col == 0:
                pdf.drawString(cursor + 2 * mm, y - 5.2 * mm, str(value))
            else:
                pdf.drawCentredString(cursor + width * mm / 2, y - 5.2 * mm, str(value))
            cursor += width * mm
        y -= 8 * mm

    y -= 10 * mm
    pdf.setFillColor(navy)
    pdf.setFont("DejaVu-Bold", 13)
    pdf.drawString(15 * mm, y, "Audit configuration")
    y -= 8 * mm
    pdf.setFillColor(slate)
    pdf.setFont("DejaVu", 9)
    pdf.drawString(15 * mm, y, (
        f"Targets: {len(settings['base_targets'])}   |   "
        f"Mutations per target: {settings['mutation_count_per_target']}   |   "
        f"Temperatures: {', '.join(str(value) for value in settings['temperatures'])}   |   "
        f"Maximum output tokens: {', '.join(str(value) for value in settings['max_tokens'])}"
    ))

    y -= 13 * mm
    pdf.setFillColor(navy)
    pdf.setFont("DejaVu-Bold", 13)
    pdf.drawString(15 * mm, y, "Observed violation tags")
    tag_counts = Counter(tag for row in payload for tag in row.get("violations", []))
    y -= 8 * mm
    pdf.setFillColor(slate)
    pdf.setFont("DejaVu", 9)
    if tag_counts:
        pdf.drawString(15 * mm, y, "  |  ".join(
            f"{tag}: {count}" for tag, count in tag_counts.most_common()))
    else:
        pdf.drawString(15 * mm, y, "No rule-based violation tags were observed.")

    y -= 14 * mm
    pdf.setFillColor(navy)
    pdf.setFont("DejaVu-Bold", 13)
    pdf.drawString(15 * mm, y, "Methodology and limitations")
    y -= 8 * mm
    pdf.setFillColor(slate)
    pdf.setFont("DejaVu", 9)
    methodology = (
        "Every provider receives the same targets, seven controlled prompt mutations, "
        "temperatures, and token limit. Violation and refusal rates use explicit rules "
        "documented in the source. Offline profiles are deterministic illustrative fixtures, "
        "not measured claims about real models. Results are screening signals, not proof of "
        "model safety, fairness, or compliance."
    )
    words = methodology.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if pdf.stringWidth(candidate, "DejaVu", 9) <= 180 * mm:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for text_line in lines:
        pdf.drawString(15 * mm, y, text_line)
        y -= 5 * mm

    pdf.setStrokeColor(line)
    pdf.line(15 * mm, 17 * mm, page_width - 15 * mm, 17 * mm)
    pdf.setFillColor(muted)
    pdf.setFont("DejaVu", 7)
    pdf.drawString(15 * mm, 11 * mm, "Generated by BlackBox Auditor")
    pdf.drawRightString(page_width - 15 * mm, 11 * mm, "Page 1")
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()
