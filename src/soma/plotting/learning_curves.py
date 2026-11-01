"""Render aggregate train/validation histories to standalone SVG."""

import argparse
import json
from pathlib import Path

from soma.paths import ROOT as WORKSPACE_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-dir", type=Path, default=WORKSPACE_ROOT / "outputs" / "history"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ROOT = args.history_dir
    W, H = 1500, 1050
    left, top, gap_x, gap_y = 78, 78, 40, 46
    cell_w, cell_h = 420, 275
    colors = {
        "train": "#3478b8",
        "validation": "#d87936",
        "train_acc": "#5b9b64",
        "val_acc": "#9a6bb5",
    }

    def txt(x, y, s, size=13, anchor="middle", weight="normal", fill="#222"):
        return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">{s}</text>'

    def line(x1, y1, x2, y2, stroke="#bbb", width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{d}/>'

    def poly(points, stroke, dash=None, width=2):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{d}" fill="none" stroke="{stroke}" stroke-width="{width}"{da}/>'

    def history(fold, phase):
        return json.loads(
            (ROOT / f"fold{fold}_{phase}_history.json").read_text(encoding="utf-8")
        )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        txt(W / 2, 34, "LOSO training and validation histories", 22, weight="bold"),
    ]
    headers = [
        "AE reconstruction (Smooth L1)",
        "SimSiam pretraining (negative cosine)",
        "MLP classifier (cross-entropy)",
    ]
    for c, h in enumerate(headers):
        svg.append(
            txt(left + c * (cell_w + gap_x) + cell_w / 2, 62, h, 16, weight="bold")
        )

    for r, fold in enumerate((1, 2, 3)):
        svg.append(
            txt(
                28,
                top + r * (cell_h + gap_y) + cell_h / 2,
                f"Fold {fold}",
                15,
                weight="bold",
            )
        )
        for c, phase in enumerate(("phase1", "phase2", "phase3")):
            x0 = left + c * (cell_w + gap_x)
            y0 = top + r * (cell_h + gap_y)
            x1, y1 = x0 + 48, y0 + 24
            pw, ph = cell_w - 68, cell_h - 66
            rows = history(fold, phase)
            epochs = [d["epoch"] for d in rows]
            max_ep = max(epochs)
            if phase == "phase1":
                series = [
                    ("train_smooth_l1", colors["train"]),
                    ("val_smooth_l1", colors["validation"]),
                ]
            elif phase == "phase2":
                series = [
                    ("train_simsiam_loss", colors["train"]),
                    ("val_simsiam_loss", colors["validation"]),
                ]
            else:
                series = [
                    ("train_ce", colors["train"]),
                    ("val_ce", colors["validation"]),
                ]
            vals = [d[k] for d in rows for k, _ in series]
            ymin, ymax = min(vals), max(vals)
            pad = max((ymax - ymin) * 0.08, 1e-4)
            ymin -= pad
            ymax += pad
            if phase == "phase3":
                ymin = min(0, ymin)
            for tick in range(5):
                gy = y1 + ph * tick / 4
                value = ymax - (ymax - ymin) * tick / 4
                svg.append(line(x1, gy, x1 + pw, gy, "#e5e5e5"))
                svg.append(
                    txt(x1 - 7, gy + 4, f"{value:.2f}", 10, anchor="end", fill="#555")
                )
            svg.append(line(x1, y1, x1, y1 + ph, "#444"))
            svg.append(line(x1, y1 + ph, x1 + pw, y1 + ph, "#444"))
            for t in range(5):
                gx = x1 + pw * t / 4
                svg.append(line(gx, y1 + ph, gx, y1 + ph + 4, "#444"))
                svg.append(
                    txt(gx, y1 + ph + 18, f"{max_ep * t / 4:.0f}", 10, fill="#555")
                )
            svg.append(txt(x1 + pw / 2, y1 + ph + 37, "Epoch", 11))
            for key, color in series:
                pts = [
                    (
                        x1 + (ep - 1) / max(1, max_ep - 1) * pw,
                        y1 + (ymax - d[key]) / (ymax - ymin) * ph,
                    )
                    for ep, d in zip(epochs, rows)
                ]
                svg.append(poly(pts, color))
            if phase == "phase3":
                sel = json.loads(
                    (ROOT / f"fold{fold}_selection.json").read_text(encoding="utf-8")
                )["selected_epoch"]
                sx = x1 + (sel - 1) / max(1, max_ep - 1) * pw
                svg.append(line(sx, y1, sx, y1 + ph, "#444", 1.4, "5,4"))
                # Accuracy is drawn on a right-hand 0-1 axis for direct inspection of selection behavior.
                for key, color in (
                    ("train_accuracy", colors["train_acc"]),
                    ("val_accuracy", colors["val_acc"]),
                ):
                    pts = [
                        (
                            x1 + (ep - 1) / max(1, max_ep - 1) * pw,
                            y1 + (1 - d[key]) * ph,
                        )
                        for ep, d in zip(epochs, rows)
                    ]
                    svg.append(poly(pts, color, "3,3", 1.5))
                for tick in range(5):
                    gy = y1 + ph * tick / 4
                    svg.append(
                        txt(
                            x1 + pw + 7,
                            gy + 4,
                            f"{1 - tick / 4:.2f}",
                            10,
                            anchor="start",
                            fill="#555",
                        )
                    )
                svg.append(
                    txt(x1 + pw / 2, y0 + 15, f"Selected epoch {sel}", 10, fill="#444")
                )
            else:
                svg.append(txt(x1 + pw + 7, y1 + 4, "", 10, anchor="start"))
            # Legend below each cell.
            legend_y = y0 + cell_h - 7
            entries = [
                ("Train loss", colors["train"], None),
                ("Validation loss", colors["validation"], None),
            ]
            if phase == "phase3":
                entries += [
                    ("Train accuracy", colors["train_acc"], "3,3"),
                    ("Validation accuracy", colors["val_acc"], "3,3"),
                ]
            start = x0 + 35
            for j, (label, color, dash) in enumerate(entries):
                lx = start + j * 95
                svg.append(
                    line(lx, legend_y - 4, lx + 18, legend_y - 4, color, 2, dash)
                )
                svg.append(txt(lx + 22, legend_y, label, 9, anchor="start"))

    svg.append(
        txt(
            W / 2,
            H - 12,
            "MLP accuracy uses the right axis; dashed vertical line marks the selected validation checkpoint.",
            12,
            fill="#444",
        )
    )
    svg.append("</svg>")
    out = args.out or ROOT / "training_validation_curves.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(svg), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
