"""Draw the corrected frozen-DAVIS figure from its checked points and metrics."""
import argparse
import csv
import json
import math
from pathlib import Path
from release_metrics import regression_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--out-pdf", type=Path, required=True)
    a = p.parse_args()
    with a.points.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    y = [float(r["y_true"]) for r in rows]
    pred = [float(r["y_pred"]) for r in rows]
    expected = json.loads(a.metrics.read_text(encoding="utf-8"))["primary_frozen"]
    m = regression_metrics(y, pred)
    for key in ("sample_count", "r2", "rmse", "ci", "r2m", "absolute_error_le_1_fraction"):
        if abs(m[key]-expected[key]) > 1e-10:
            raise ValueError("Plot/table metric mismatch: " + key)
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor, Color
    a.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(a.out_pdf), pagesize=(300, 325), invariant=1, pageCompression=1)
    c.setTitle("Supplementary Figure S2. Frozen DAVIS predictions")
    c.setAuthor("ConfMux-DTA")
    left, bottom, side = 47, 48, 230
    lo = math.floor(min(min(y), min(pred)))
    hi = math.ceil(max(max(y), max(pred)))
    def pos(v):
        return (v-lo)/(hi-lo)*side
    c.setFillColor(HexColor("#172331"))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left, 309, "DAVIS: frozen predictions")
    c.setFont("Helvetica", 8)
    c.drawString(left, 296, "No calibration fitted to DAVIS labels")
    c.saveState()
    path = c.beginPath()
    path.rect(left, bottom, side, side)
    c.clipPath(path, stroke=0)
    c.setFillColor(HexColor("#226AAA"))
    c.setFillAlpha(0.22)
    for observed, prediction in zip(y, pred):
        c.circle(left+pos(observed), bottom+pos(prediction), 0.55, fill=1, stroke=0)
    c.setFillAlpha(1)
    c.setStrokeColor(HexColor("#5A626B"))
    c.setLineWidth(0.75)
    c.setDash(3, 2)
    c.line(left, bottom, left+side, bottom+side)
    c.restoreState()
    c.setStrokeColor(HexColor("#26313E"))
    c.setLineWidth(0.7)
    c.rect(left, bottom, side, side, fill=0, stroke=1)
    c.setFillColor(HexColor("#26313E"))
    c.setFont("Helvetica", 8)
    for v in range(lo, hi+1):
        x = left+pos(v)
        yy = bottom+pos(v)
        c.line(x, bottom, x, bottom-3)
        c.line(left-3, yy, left, yy)
        c.drawCentredString(x, bottom-14, str(v))
        c.drawRightString(left-7, yy-2.5, str(v))
    c.setFont("Helvetica", 9)
    c.drawCentredString(left+side/2, 18, "Observed pKd")
    c.saveState()
    c.translate(13, bottom+side/2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Predicted pKd")
    c.restoreState()
    c.setFillColor(Color(1, 1, 1, alpha=0.95))
    c.rect(left+6, bottom+side-79, 101, 72, stroke=0, fill=1)
    c.setFillColor(HexColor("#172331"))
    c.setFont("Helvetica", 8)
    labels = [f"N = {len(y):,}", f"R² = {m['r2']:.3f}", f"RMSE = {m['rmse']:.3f}",
              f"CI = {m['ci']:.3f}", f"Rm² = {m['r2m']:.3f}",
              f"|error| <= 1: {100*m['absolute_error_le_1_fraction']:.1f}%"]
    for i, text in enumerate(labels):
        c.drawString(left+11, bottom+side-18-i*10, text)
    c.showPage()
    c.save()
    print(a.out_pdf)


if __name__ == "__main__":
    main()

