"""Plot the VP fixed-speculative token/patch sweep and selected finalists."""

import argparse
import csv
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "baseline": "#3274B4",
    "speculative": "#B65A6A",
    "spec_token": "#DF9F2D",
    "full_stack": "#4B9B69",
    "selected": "#7554B8",
    "text": "#172033",
    "muted": "#5B6472",
    "grid": "#E2E8F0",
}


def font(size, bold=False):
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else (
        "arial.ttf", "DejaVuSans.ttf"
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def long_path(path):
    value = str(Path(path).resolve())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return value


def read_rows(path):
    with open(long_path(path), newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for row in rows:
        if row.get("status") != "complete":
            continue
        try:
            row["mae"] = float(row["mae"])
            row["latency_mean_ms"] = float(row["latency_mean_ms"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(row["mae"]) and math.isfinite(row["latency_mean_ms"]):
            output.append(row)
    if not output:
        raise ValueError(f"no completed rows: {path}")
    return output


def patch_label(row):
    case = row.get("patch_case") or row["case"].split("_token_")[0].removeprefix("full_")
    return case.replace("gated_adaptive", "adaptive").replace("_cache", "").replace("_", " ")


def token_label(row):
    if row["family"] == "speculative":
        return "Spec only (K=10 tokens)"
    return f"Token K={int(float(row['recent_k']))}"


def bounds(values, padding=0.12):
    low, high = min(values), max(values)
    span = high - low or max(abs(high), 1.0) * 0.03
    return low - span * padding, high + span * padding


def dot_panel(draw, rows, labels, selected, box, metric, title, digits):
    left, top, right, bottom = box
    label_right = left + 245
    axis_left, axis_right = label_right + 18, right - 90
    values = [row[metric] for row in rows]
    low, high = bounds(values)
    row_height = (bottom - top - 42) / max(len(rows), 1)
    draw.text(((axis_left + axis_right) / 2, top + 12), title, anchor="mm",
              font=font(15, True), fill=COLORS["text"])
    for step in range(5):
        value = low + (high - low) * step / 4
        x = axis_left + (value - low) / (high - low) * (axis_right - axis_left)
        draw.line((x, top + 35, x, bottom), fill=COLORS["grid"], width=1)
        draw.text((x, bottom + 4), f"{value:.{digits}f}", anchor="ma",
                  font=font(9), fill=COLORS["muted"])
    for index, (row, label) in enumerate(zip(rows, labels)):
        y = top + 48 + index * row_height
        chosen = row["case"] in selected
        draw.text((label_right, y), label, anchor="rm", font=font(10, chosen),
                  fill=COLORS["text"])
        draw.line((axis_left, y, axis_right, y), fill=COLORS["grid"], width=2)
        x = axis_left + (row[metric] - low) / (high - low) * (axis_right - axis_left)
        color = COLORS["selected"] if chosen else COLORS[row["family"]]
        radius = 7 if chosen else 5
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color,
                     outline="#111827" if chosen else "white", width=2)
        draw.text((x + 10, y), f"{row[metric]:.{digits}f}", anchor="lm",
                  font=font(9, chosen), fill=COLORS["text"])


def plot_sweep(rows, final_rows, output):
    selected = {row["case"] for row in final_rows}
    spec = next(row for row in rows if row["family"] == "speculative")
    token = [spec] + sorted(
        (row for row in rows if row["family"] == "spec_token"),
        key=lambda row: float(row["recent_k"]),
    )
    patches = sorted(
        (row for row in rows if row["family"] == "full_stack"),
        key=lambda row: (row["policy"], float(row["threshold"]), int(row["max_skip"])),
    )
    image = Image.new("RGB", (1900, 1120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((38, 34), "VP fixed-speculative Token and Patch sweep", anchor="lm",
              font=font(26, True), fill=COLORS["text"])
    draw.text((38, 68), "Speculative fixed at G=6, T=0.4. Purple points are final selected settings.",
              anchor="lm", font=font(13), fill=COLORS["muted"])
    draw.text((38, 108), "1. Token sweep (Spec fixed)", anchor="lm",
              font=font(19, True), fill=COLORS["text"])
    token_labels = [token_label(row) for row in token]
    dot_panel(draw, token, token_labels, selected, (25, 130, 940, 405),
              "mae", "MAE (deg; lower is better)", 3)
    dot_panel(draw, token, token_labels, selected, (950, 130, 1875, 405),
              "latency_mean_ms", "Mean latency (ms; lower is better)", 2)
    draw.text((38, 470), "2. Patch sweep (Spec G=6,T=0.4 + Token K=8 fixed)",
              anchor="lm", font=font(19, True), fill=COLORS["text"])
    patch_labels = [patch_label(row) for row in patches]
    dot_panel(draw, patches, patch_labels, selected, (25, 495, 940, 1020),
              "mae", "MAE (deg; lower is better)", 3)
    dot_panel(draw, patches, patch_labels, selected, (950, 495, 1875, 1020),
              "latency_mean_ms", "Mean latency (ms; lower is better)", 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(long_path(output), "PNG", optimize=True)


def final_label(row):
    if row["family"] == "baseline":
        return "NBS compact only"
    if row["family"] == "speculative":
        return "Spec G6/T0.4 only"
    if row["family"] == "spec_token":
        return f"Spec + Token K={int(float(row['recent_k']))}"
    return "Full stack: " + patch_label(row)


def bar_panel(draw, rows, box, metric, title, digits):
    left, top, right, bottom = box
    maximum = max(row[metric] for row in rows) * 1.12
    row_height = (bottom - top) / len(rows)
    draw.text(((left + right) / 2, top - 34), title, anchor="mm",
              font=font(16, True), fill=COLORS["text"])
    for index, row in enumerate(rows):
        y = top + index * row_height
        width = (right - left) * row[metric] / maximum
        draw.rounded_rectangle((left, y+5, left+width, y+32), 4,
                               fill=COLORS.get(row["family"], COLORS["selected"]))
        draw.text((left+width+8, y+18), f"{row[metric]:.{digits}f}", anchor="lm",
                  font=font(11), fill=COLORS["text"])


def plot_final(rows, output):
    width, top, row_height = 1800, 150, 62
    height = top + len(rows) * row_height + 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((38, 34), "VP selected settings under fixed speculative decoding",
              anchor="lm", font=font(25, True), fill=COLORS["text"])
    draw.text((38, 68), "Final comparison: baseline, fixed Spec, selected Token, quality and speed Patch finalists",
              anchor="lm", font=font(13), fill=COLORS["muted"])
    for index, row in enumerate(rows):
        draw.text((430, top + index*row_height + 18), final_label(row), anchor="rm",
                  font=font(11, row["family"] == "full_stack"), fill=COLORS["text"])
    bottom = top + len(rows)*row_height
    bar_panel(draw, rows, (460, top, 970, bottom), "mae", "MAE (deg; lower is better)", 3)
    bar_panel(draw, rows, (1160, top, 1670, bottom), "latency_mean_ms",
              "Mean latency (ms; lower is better)", 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(long_path(output), "PNG", optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    output = (args.output_dir or root / "plots").resolve()
    rows = read_rows(root / "fixed_spec_patch_token_sweep.csv")
    final_rows = read_rows(root / "fixed_spec_final_comparison.csv")
    plot_sweep(rows, final_rows, output / "fixed_spec_parameter_sweeps.png")
    plot_final(final_rows, output / "fixed_spec_best_combinations.png")
    print(f"Saved: {output / 'fixed_spec_parameter_sweeps.png'}")
    print(f"Saved: {output / 'fixed_spec_best_combinations.png'}")


if __name__ == "__main__":
    main()
