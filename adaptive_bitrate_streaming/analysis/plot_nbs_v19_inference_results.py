"""Create PNG charts from the NBS v19 inference matrix CSV."""

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    'qoe': '#2f855a',
    'latency': '#c05621',
    'baseline': '#2b6cb0',
    'official': '#6b46c1',
    'grid': '#d1d5db',
    'text': '#172033',
    'muted': '#5b6472',
    'background': '#ffffff',
}


def load_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'CSV contains no result rows: {path}')
    required = ('experiment', 'mean_reward', 'inference_latency_mean_ms')
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise ValueError(f'CSV is missing columns: {", ".join(missing)}')
    for row in rows:
        for field in required[1:]:
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError(f'{row["experiment"]}: {field} is not finite')
    return rows


def append_netllm_metrics(rows, path):
    with path.open(encoding='utf-8') as stream:
        metrics = json.load(stream)
    row = {
        'experiment': 'netllm_official_lora',
        'mean_reward': float(metrics['mean_reward']),
        'inference_latency_mean_ms': float(
            metrics['inference_latency_mean_ms']
        ),
    }
    if not all(math.isfinite(row[field]) for field in (
        'mean_reward', 'inference_latency_mean_ms'
    )):
        raise ValueError(f'official NetLLM metrics are not finite: {path}')
    return [*rows, row]


def bar_color(row, metric):
    if row['experiment'] == 'nbs_only':
        return COLORS['baseline']
    if row['experiment'] == 'netllm_official_lora':
        return COLORS['official']
    return COLORS['qoe'] if metric == 'mean_reward' else COLORS['latency']


def font(size, bold=False):
    candidates = (
        ('arialbd.ttf', 'DejaVuSans-Bold.ttf') if bold
        else ('arial.ttf', 'DejaVuSans.ttf')
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text(draw, position, value, anchor='la', size=13, bold=False, color='text'):
    draw.text(
        position, str(value), anchor=anchor, font=font(size, bold),
        fill=COLORS[color],
    )


def horizontal_chart(rows, metric, title, output_path, ascending=False):
    ordered = sorted(rows, key=lambda row: row[metric], reverse=not ascending)
    width = 1200
    left = 365
    right = 105
    top = 92
    row_height = 48
    height = top + row_height * len(ordered) + 65
    values = [row[metric] for row in ordered]
    maximum = max(values) * 1.12 if max(values) > 0 else 1.0
    chart_width = width - left - right
    baseline = next((row[metric] for row in rows if row['experiment'] == 'nbs_only'), None)
    unit = 'QoE' if metric == 'mean_reward' else 'ms/call'
    image = Image.new('RGB', (width, height), COLORS['background'])
    draw = ImageDraw.Draw(image)
    text(draw, (32, 38), title, anchor='lm', size=22, bold=True)
    text(
        draw, (32, 64),
        f'NBS-only (blue) and official NetLLM LoRA (purple) · {unit}',
        anchor='lm', size=13, color='muted',
    )
    if baseline is not None:
        baseline_x = left + chart_width * baseline / maximum
        draw.line(
            (baseline_x, top - 16, baseline_x, height - 48),
            fill=COLORS['baseline'], width=2,
        )
    for index, row in enumerate(ordered):
        y = top + index * row_height
        bar_width = chart_width * row[metric] / maximum
        color = bar_color(row, metric)
        text(draw, (left - 14, y + 14), row['experiment'], anchor='rm')
        draw.rounded_rectangle(
            (left, y, left + bar_width, y + 28), radius=3, fill=color
        )
        value = (
            f'{row[metric]:.3f}' if metric == 'mean_reward'
            else f'{row[metric]:.1f}'
        )
        text(draw, (left + bar_width + 9, y + 14), value, anchor='lm', size=12)
    draw.line(
        (left, height - 46, width - right, height - 46),
        fill=COLORS['grid'], width=1,
    )
    text(
        draw, ((left + width - right) / 2, height - 18), unit,
        anchor='mm', color='muted',
    )
    image.save(output_path, format='PNG', optimize=True)


def overview_chart(rows, output_path):
    width = 1440
    label_width = 315
    panel_width = 430
    panel_gap = 130
    left_one = label_width + 25
    left_two = left_one + panel_width + panel_gap
    top = 112
    row_height = 48
    height = top + row_height * len(rows) + 70
    qoe_max = max(row['mean_reward'] for row in rows) * 1.14
    latency_max = max(row['inference_latency_mean_ms'] for row in rows) * 1.14
    image = Image.new('RGB', (width, height), COLORS['background'])
    draw = ImageDraw.Draw(image)
    text(
        draw, (32, 38), 'NBS v19 inference overview',
        anchor='lm', size=22, bold=True,
    )
    text(
        draw, (32, 64),
        'Original order · NBS-only blue, official NetLLM LoRA purple',
        anchor='lm', size=13, color='muted',
    )
    text(
        draw, (left_one + panel_width / 2, 94), 'QoE ↑',
        anchor='mm', bold=True,
    )
    text(
        draw, (left_two + panel_width / 2, 94),
        'Mean inference latency (ms) ↓', anchor='mm', bold=True,
    )
    for index, row in enumerate(rows):
        y = top + index * row_height
        qoe_color = bar_color(row, 'mean_reward')
        latency_color = bar_color(row, 'inference_latency_mean_ms')
        qoe_width = panel_width * row['mean_reward'] / qoe_max
        latency_width = panel_width * row['inference_latency_mean_ms'] / latency_max
        text(draw, (label_width, y + 14), row['experiment'], anchor='rm')
        draw.rounded_rectangle(
            (left_one, y, left_one + qoe_width, y + 28),
            radius=3, fill=qoe_color,
        )
        text(
            draw, (left_one + qoe_width + 8, y + 14),
            f'{row["mean_reward"]:.3f}', anchor='lm', size=12,
        )
        draw.rounded_rectangle(
            (left_two, y, left_two + latency_width, y + 28),
            radius=3, fill=latency_color,
        )
        text(
            draw, (left_two + latency_width + 8, y + 14),
            f'{row["inference_latency_mean_ms"]:.1f}', anchor='lm', size=12,
        )
    image.save(output_path, format='PNG', optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=Path)
    parser.add_argument(
        '--netllm-metrics-json', type=Path,
        help='selector_metrics.json produced by the official rank-128 NetLLM LoRA',
    )
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--prefix', default='nbs_v19')
    args = parser.parse_args()
    output_dir = args.output_dir or args.csv_path.parent / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.csv_path)
    if args.netllm_metrics_json is not None:
        rows = append_netllm_metrics(rows, args.netllm_metrics_json)
    outputs = (
        output_dir / f'{args.prefix}_overview.png',
        output_dir / f'{args.prefix}_qoe_sorted.png',
        output_dir / f'{args.prefix}_latency_sorted.png',
    )
    overview_chart(rows, outputs[0])
    horizontal_chart(
        rows, 'mean_reward', 'QoE ranking', outputs[1], ascending=False
    )
    horizontal_chart(
        rows, 'inference_latency_mean_ms', 'Inference latency ranking',
        outputs[2], ascending=True,
    )
    for path in outputs:
        print(path.resolve())


if __name__ == '__main__':
    main()
