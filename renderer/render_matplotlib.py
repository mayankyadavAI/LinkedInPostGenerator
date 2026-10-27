#!/usr/bin/env python3
"""
Matplotlib diagram renderer — 6 layout types, no browser needed.

Dispatches to the right drawing function based on content["diagram_type"]:
  flow    — pipeline / step-by-step process
  split   — comparison / before vs after
  layers  — hierarchy / stack
  cycle   — feedback loop / repeating cycle
  scale   — spectrum / tradeoff axis
  stats   — large number callouts

Usage:
    from renderer.render_matplotlib import render
    render(content_dict, "output/infographic.png")

Standalone test:
    python renderer/render_matplotlib.py [diagram_type]
"""

import math
import pathlib
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── canvas ─────────────────────────────────────────────────────────────────────
W_PX, H_PX = 1080, 1350
DPI = 150
W_IN, H_IN = W_PX / DPI, H_PX / DPI

# ── dark palette ───────────────────────────────────────────────────────────────
BG      = "#060b14"
SURF    = "#0d1520"
SURF2   = "#141f2e"
SURF3   = "#1a2740"
BORDER  = "#1e2d42"
INK     = "#edf0f7"
INK2    = "#8a9ab8"
INK3    = "#4a566a"
BLUE    = "#4f8dff"
VIOLET  = "#9d6fff"
TEAL    = "#2ec4b6"
AMBER   = "#f5a623"
ROSE    = "#f06292"
GREEN   = "#34d399"
ACCENTS = [BLUE, VIOLET, TEAL, AMBER, ROSE, GREEN]


# ── shared helpers ─────────────────────────────────────────────────────────────

def _fig():
    fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _top_bar(ax, colors=None):
    colors = colors or ACCENTS[:4]
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle(
            (i / len(colors), 0.994), 1 / len(colors), 0.006,
            facecolor=c, transform=ax.transAxes, zorder=10, clip_on=False))


def _bottom_bar(ax, colors=None):
    colors = colors or ACCENTS[:4]
    for i, c in enumerate(colors):
        ax.add_patch(mpatches.Rectangle(
            (i / len(colors), 0), 1 / len(colors), 0.005,
            facecolor=c, transform=ax.transAxes, zorder=10, clip_on=False))


def _box(ax, x, y, w, h, fc=SURF, ec=BORDER, lw=1.0, radius=0.02, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=ax.transAxes, zorder=zorder, clip_on=False))


def _txt(ax, x, y, s, size=9, color=INK, weight="normal",
         ha="center", va="center", alpha=1.0, zorder=5):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, alpha=alpha, transform=ax.transAxes,
            zorder=zorder, fontfamily="DejaVu Sans", clip_on=False)


def _wrap_txt(ax, x, y, s, size=9, color=INK, weight="normal",
              ha="center", width=40, line_gap=None, zorder=5):
    """Render wrapped text, returns y after last line."""
    if line_gap is None:
        line_gap = size * 0.0013
    lines = textwrap.fill(s, width=width).split("\n")
    for i, line in enumerate(lines):
        _txt(ax, x, y - i * line_gap, line, size=size,
             color=color, weight=weight, ha=ha, va="top", zorder=zorder)
    return y - (len(lines) - 1) * line_gap


def _arrow_h(ax, x0, x1, y, color=INK3, lw=1.2):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=6)


def _arrow_v(ax, x, y0, y1, color=INK3, lw=1.2):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=6)


def _header(ax, title, subtitle, accent=VIOLET):
    """Draw title + subtitle block, returns y where body should start."""
    _txt(ax, 0.5, 0.940, title,
         size=19, color=INK, weight="bold", ha="center")
    _txt(ax, 0.5, 0.900, subtitle,
         size=9.5, color=INK2, ha="center")
    # thin divider
    ax.plot([0.05, 0.95], [0.880, 0.880],
            color=accent, linewidth=0.8, alpha=0.4,
            transform=ax.transAxes, zorder=1)
    return 0.870


def _takeaway(ax, text, y=0.090):
    """Draw bottom takeaway box."""
    _box(ax, 0.05, y - 0.010, 0.90, 0.058, fc=SURF2, ec=VIOLET, lw=1.0)
    ax.add_patch(mpatches.Rectangle(
        (0.05, y - 0.010), 0.007, 0.058,
        facecolor=VIOLET, transform=ax.transAxes, zorder=4, clip_on=False))
    _wrap_txt(ax, 0.52, y + 0.030, text,
              size=8.5, color=INK, ha="center", width=72)


def _save(fig, out_path):
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  [render] → {out_path}")
    return str(out_path)


# ── LAYOUT 1: flow ─────────────────────────────────────────────────────────────

def _render_flow(content: dict, out_path: str) -> str:
    steps = content.get("steps", [])[:5]
    n = len(steps)
    colors = ACCENTS[:n]

    fig, ax = _fig()
    _top_bar(ax, colors)
    _bottom_bar(ax, colors)

    body_top = _header(ax, content.get("title", ""), content.get("subtitle", ""),
                       accent=BLUE)

    # vertical stack of step boxes with arrows between
    box_h = 0.105
    gap = 0.030
    total = n * box_h + (n - 1) * gap
    y_start = body_top - 0.02 - (body_top - 0.02 - 0.175 - total) / 2

    for i, step in enumerate(steps):
        col = colors[i]
        y = y_start - i * (box_h + gap)
        _box(ax, 0.08, y - box_h, 0.84, box_h,
             fc=SURF, ec=col, lw=1.4, radius=0.015)
        # left accent strip
        ax.add_patch(mpatches.Rectangle(
            (0.08, y - box_h), 0.010, box_h,
            facecolor=col, transform=ax.transAxes, zorder=4, clip_on=False))
        # step number
        _txt(ax, 0.120, y - box_h / 2, str(i + 1),
             size=13, color=col, weight="bold", ha="center")
        # label
        _txt(ax, 0.580, y - box_h * 0.28,
             step.get("label", ""),
             size=10, color=INK, weight="bold", ha="center")
        # detail
        _wrap_txt(ax, 0.580, y - box_h * 0.60,
                  step.get("detail", ""),
                  size=8.5, color=INK2, ha="center", width=55)
        # arrow down (not after last)
        if i < n - 1:
            _arrow_v(ax, 0.5, y - box_h - 0.002,
                     y - box_h - gap + 0.004, color=col)

    _takeaway(ax, content.get("takeaway", ""))
    return _save(fig, out_path)


# ── LAYOUT 2: split ────────────────────────────────────────────────────────────

def _render_split(content: dict, out_path: str) -> str:
    rows = content.get("rows", [])[:5]

    fig, ax = _fig()
    _top_bar(ax, [ROSE, TEAL, ROSE, TEAL])
    _bottom_bar(ax, [ROSE, TEAL, ROSE, TEAL])

    body_top = _header(ax, content.get("title", ""),
                       content.get("subtitle", ""), accent=TEAL)

    # column headers
    _box(ax, 0.05, body_top - 0.065, 0.41, 0.048,
         fc="#1a0a0a", ec=ROSE, lw=1.4, radius=0.012)
    _txt(ax, 0.255, body_top - 0.040,
         content.get("left_label", "BEFORE"),
         size=10, color=ROSE, weight="bold")

    _box(ax, 0.54, body_top - 0.065, 0.41, 0.048,
         fc="#0a1a12", ec=TEAL, lw=1.4, radius=0.012)
    _txt(ax, 0.745, body_top - 0.040,
         content.get("right_label", "AFTER"),
         size=10, color=TEAL, weight="bold")

    # VS divider
    _txt(ax, 0.500, body_top - 0.040, "VS",
         size=9, color=INK3, weight="bold")

    row_h = 0.088
    row_gap = 0.012
    y0 = body_top - 0.085

    for i, row in enumerate(rows):
        y = y0 - i * (row_h + row_gap)
        # left cell
        _box(ax, 0.05, y - row_h, 0.41, row_h,
             fc=SURF, ec="#3a1a1a", lw=0.8)
        _wrap_txt(ax, 0.255, y - row_h * 0.35,
                  row.get("left", ""), size=8.5,
                  color="#f8a0a0", ha="center", width=28)
        # right cell
        _box(ax, 0.54, y - row_h, 0.41, row_h,
             fc=SURF, ec="#0e2a20", lw=0.8)
        _wrap_txt(ax, 0.745, y - row_h * 0.35,
                  row.get("right", ""), size=8.5,
                  color="#6ee7b7", ha="center", width=28)
        # centre line
        ax.plot([0.5, 0.5], [y - row_h, y],
                   color=BORDER, linewidth=0.6,
                   transform=ax.transAxes, zorder=1)

    _takeaway(ax, content.get("takeaway", ""))
    return _save(fig, out_path)


# ── LAYOUT 3: layers ───────────────────────────────────────────────────────────

def _render_layers(content: dict, out_path: str) -> str:
    layers = content.get("layers", [])[:6]
    n = len(layers)
    colors = [BLUE, VIOLET, TEAL, AMBER, ROSE, GREEN][:n]

    fig, ax = _fig()
    _top_bar(ax, colors)
    _bottom_bar(ax, colors)

    body_top = _header(ax, content.get("title", ""),
                       content.get("subtitle", ""), accent=AMBER)

    # Draw layers as a pyramid — widest at bottom, narrowest at top
    layer_h = 0.090
    gap = 0.008
    total = n * layer_h + (n - 1) * gap
    y_start = body_top - 0.02

    for i, layer in enumerate(layers):
        col = colors[i]
        # each layer gets narrower going up (index 0 = top = narrow)
        indent = (n - 1 - i) * 0.04   # bottom layers wider
        x = 0.05 + indent
        w = 0.90 - 2 * indent
        y = y_start - i * (layer_h + gap) - layer_h

        _box(ax, x, y, w, layer_h, fc=SURF2, ec=col, lw=1.3, radius=0.012)
        # left label badge
        ax.add_patch(mpatches.Rectangle(
            (x, y), 0.008, layer_h,
            facecolor=col, transform=ax.transAxes, zorder=4, clip_on=False))
        # layer number
        _txt(ax, x + 0.030, y + layer_h / 2, str(i + 1),
             size=8, color=col, weight="bold")
        # label
        _txt(ax, x + w * 0.38, y + layer_h * 0.65,
             layer.get("label", ""),
             size=9.5, color=INK, weight="bold", ha="center")
        # detail
        _wrap_txt(ax, x + w * 0.38, y + layer_h * 0.32,
                  layer.get("detail", ""),
                  size=8, color=INK2, ha="center", width=40)

    _takeaway(ax, content.get("takeaway", ""))
    return _save(fig, out_path)


# ── LAYOUT 4: cycle ────────────────────────────────────────────────────────────

def _render_cycle(content: dict, out_path: str) -> str:
    steps = content.get("steps", [])[:6]
    n = len(steps)
    colors = ACCENTS[:n]

    fig, ax = _fig()
    _top_bar(ax, colors)
    _bottom_bar(ax, colors)

    body_top = _header(ax, content.get("title", ""),
                       content.get("subtitle", ""), accent=TEAL)

    # Draw nodes in a circle
    cx, cy = 0.50, 0.490
    radius = 0.280
    box_w, box_h = 0.230, 0.095

    angles = [math.pi / 2 + 2 * math.pi * i / n for i in range(n)]
    node_centers = [
        (cx + radius * math.cos(a), cy + radius * math.sin(a))
        for a in angles
    ]

    # Draw curved arrows between nodes
    for i in range(n):
        x0, y0 = node_centers[i]
        x1, y1 = node_centers[(i + 1) % n]
        col = colors[i]
        # midpoint slightly towards center for curve effect
        mx = (x0 + x1) / 2 * 0.95 + cx * 0.05
        my = (y0 + y1) / 2 * 0.95 + cy * 0.05
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="-|>", color=col, lw=1.3,
                        connectionstyle=f"arc3,rad=0.25"),
                    zorder=3)

    # Draw step boxes
    for i, (step, (nx, ny), col) in enumerate(zip(steps, node_centers, colors)):
        _box(ax, nx - box_w / 2, ny - box_h / 2, box_w, box_h,
             fc=SURF2, ec=col, lw=1.4, radius=0.015, zorder=5)
        _txt(ax, nx, ny + box_h * 0.18,
             step.get("label", ""),
             size=8.5, color=INK, weight="bold")
        _wrap_txt(ax, nx, ny - box_h * 0.12,
                  step.get("detail", ""),
                  size=7.5, color=INK2, ha="center", width=22)

    # Centre label
    _box(ax, cx - 0.12, cy - 0.055, 0.24, 0.110,
         fc=BG, ec=BORDER, lw=0.8, radius=0.020, zorder=4)
    _txt(ax, cx, cy + 0.020, "CYCLE", size=8, color=INK3, weight="bold")
    _txt(ax, cx, cy - 0.015, "↻", size=18, color=VIOLET, weight="bold")

    _takeaway(ax, content.get("takeaway", ""))
    return _save(fig, out_path)


# ── LAYOUT 5: scale ────────────────────────────────────────────────────────────

def _render_scale(content: dict, out_path: str) -> str:
    markers = content.get("markers", [])[:4]

    fig, ax = _fig()
    _top_bar(ax, [BLUE, VIOLET, AMBER])
    _bottom_bar(ax, [BLUE, VIOLET, AMBER])

    body_top = _header(ax, content.get("title", ""),
                       content.get("subtitle", ""), accent=VIOLET)

    # Axis bar
    axis_y = 0.540
    ax_left, ax_right = 0.08, 0.92
    ax_width = ax_right - ax_left

    # gradient bar: blue → violet → amber
    n_segs = 60
    for i in range(n_segs):
        t = i / n_segs
        # interpolate BLUE→VIOLET→AMBER
        if t < 0.5:
            r = int(0x4f + (0x9d - 0x4f) * t * 2)
            g = int(0x8d + (0x6f - 0x8d) * t * 2)
            b = int(0xff + (0xff - 0xff) * t * 2)
        else:
            t2 = (t - 0.5) * 2
            r = int(0x9d + (0xf5 - 0x9d) * t2)
            g = int(0x6f + (0xa6 - 0x6f) * t2)
            b = int(0xff + (0x23 - 0xff) * t2)
        color = f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"
        seg_x = ax_left + i / n_segs * ax_width
        ax.add_patch(mpatches.Rectangle(
            (seg_x, axis_y - 0.012), ax_width / n_segs + 0.001, 0.024,
            facecolor=color, transform=ax.transAxes, zorder=3, clip_on=False))

    # pole labels
    _txt(ax, ax_left, axis_y - 0.040,
         content.get("left_pole", "Low"),
         size=9, color=BLUE, weight="bold", ha="left")
    _txt(ax, ax_right, axis_y - 0.040,
         content.get("right_pole", "High"),
         size=9, color=AMBER, weight="bold", ha="right")
    _txt(ax, 0.5, axis_y - 0.060,
         content.get("axis_label", ""),
         size=8, color=INK3, ha="center")

    # markers above/below alternating
    for i, m in enumerate(markers):
        pos = float(m.get("position", 0.5))
        mx = ax_left + pos * ax_width
        above = i % 2 == 0
        # stem
        stem_y0 = axis_y + 0.012 if above else axis_y - 0.012
        stem_y1 = axis_y + 0.090 if above else axis_y - 0.090
        ax.plot([mx, mx], [stem_y0, stem_y1], color=INK3,
                lw=0.8, transform=ax.transAxes, zorder=2)
        # dot on axis
        ax.plot(mx, axis_y, "o", ms=7, color=INK,
                transform=ax.transAxes, zorder=5)
        # label box
        box_y = stem_y1 + 0.005 if above else stem_y1 - 0.075
        _box(ax, mx - 0.110, box_y, 0.220, 0.065,
             fc=SURF2, ec=VIOLET, lw=0.9, radius=0.012)
        _txt(ax, mx, box_y + 0.043,
             m.get("label", ""),
             size=8.5, color=INK, weight="bold")
        _wrap_txt(ax, mx, box_y + 0.022,
                  m.get("detail", ""),
                  size=7.5, color=INK2, ha="center", width=24)

    _takeaway(ax, content.get("takeaway", ""))
    return _save(fig, out_path)


# ── LAYOUT 6: stats ────────────────────────────────────────────────────────────

def _render_stats(content: dict, out_path: str) -> str:
    stats = content.get("stats", [])[:4]
    n = len(stats)
    colors = [TEAL, VIOLET, AMBER, BLUE][:n]

    fig, ax = _fig()
    _top_bar(ax, colors)
    _bottom_bar(ax, colors)

    body_top = _header(ax, content.get("title", ""),
                       content.get("subtitle", ""), accent=TEAL)

    # arrange in 2×2 or 1×n grid
    cols = 2 if n > 2 else n
    rows = math.ceil(n / cols)
    cell_w = 0.82 / cols
    cell_h = 0.170
    pad_x = (1 - cols * cell_w) / 2
    y0 = body_top - 0.04

    for i, (stat, col) in enumerate(zip(stats, colors)):
        row_i = i // cols
        col_i = i % cols
        sx = pad_x + col_i * (cell_w + 0.04)
        sy = y0 - row_i * (cell_h + 0.030) - cell_h

        _box(ax, sx, sy, cell_w, cell_h,
             fc=SURF, ec=col, lw=1.5, radius=0.018, zorder=2)
        # top colour strip
        ax.add_patch(mpatches.Rectangle(
            (sx, sy + cell_h - 0.007), cell_w, 0.007,
            facecolor=col, transform=ax.transAxes, zorder=4, clip_on=False))

        # big value
        _txt(ax, sx + cell_w / 2, sy + cell_h * 0.63,
             stat.get("value", "?"),
             size=22, color=col, weight="bold")
        # label
        _txt(ax, sx + cell_w / 2, sy + cell_h * 0.35,
             stat.get("label", ""),
             size=9, color=INK, weight="bold")
        # detail
        _wrap_txt(ax, sx + cell_w / 2, sy + cell_h * 0.16,
                  stat.get("detail", ""),
                  size=7.5, color=INK2, ha="center", width=28)

    _takeaway(ax, content.get("takeaway", ""))
    return _save(fig, out_path)


# ── dispatch ───────────────────────────────────────────────────────────────────

_RENDERERS = {
    "flow":   _render_flow,
    "split":  _render_split,
    "layers": _render_layers,
    "cycle":  _render_cycle,
    "scale":  _render_scale,
    "stats":  _render_stats,
}


def render(content: dict, out_path: str) -> str:
    """Render content dict → PNG. Dispatches by content['diagram_type']."""
    dtype = content.get("diagram_type", "flow")
    fn = _RENDERERS.get(dtype, _render_flow)
    print(f"  [render] diagram_type={dtype}")
    return fn(content, out_path)


# ── test samples ───────────────────────────────────────────────────────────────

_SAMPLES = {
    "flow": {
        "diagram_type": "flow",
        "title": "How an LLM Predicts the Next Word",
        "subtitle": "From your prompt to the output token — step by step",
        "steps": [
            {"label": "Tokenise Input",   "detail": "Your text is split into tokens, not words"},
            {"label": "Embed Tokens",     "detail": "Each token becomes a vector of numbers"},
            {"label": "Run Attention",    "detail": "Every token looks at every other token"},
            {"label": "Predict Logits",   "detail": "The model scores every possible next token"},
            {"label": "Sample Output",    "detail": "One token is picked — then the loop repeats"},
        ],
        "takeaway": "There's no lookup table — the model is guessing the most likely next token every single time.",
    },
    "split": {
        "diagram_type": "split",
        "title": "What People Think vs What AI Actually Does",
        "subtitle": "The gap between the mental model and the reality",
        "left_label": "Common Belief",
        "right_label": "What's Actually True",
        "rows": [
            {"left": "AI searches the web",       "right": "Predicts from training data"},
            {"left": "It knows facts",             "right": "It learned word patterns"},
            {"left": "Confident = correct",        "right": "Confidence is a side effect"},
            {"left": "It understands meaning",     "right": "It matches statistical patterns"},
        ],
        "takeaway": "LLMs don't retrieve — they predict. That's why they sound right even when they're wrong.",
    },
    "layers": {
        "diagram_type": "layers",
        "title": "The AI Stack, From Chips to Chatbot",
        "subtitle": "What actually runs under every AI response",
        "layers": [
            {"label": "Your Prompt",      "detail": "What you type → tokenised and sent in"},
            {"label": "LLM Inference",    "detail": "Model runs forward pass, outputs logits"},
            {"label": "CUDA Kernels",     "detail": "Matrix multiplications on GPU cores"},
            {"label": "GPU Memory",       "detail": "Weights + KV cache loaded in VRAM"},
            {"label": "Hardware",         "detail": "H100 / A100 / M-series silicon"},
        ],
        "takeaway": "Every chat message triggers billions of floating-point ops across layers of abstraction.",
    },
    "cycle": {
        "diagram_type": "cycle",
        "title": "The RLHF Feedback Loop",
        "subtitle": "How human ratings shape model behaviour",
        "steps": [
            {"label": "Generate Output",   "detail": "Model produces several responses"},
            {"label": "Human Rates",       "detail": "Humans rank best to worst"},
            {"label": "Train Reward Model","detail": "Learns what humans prefer"},
            {"label": "Update Policy",     "detail": "Model adjusts toward higher reward"},
        ],
        "takeaway": "RLHF is how 'sounds helpful' gets baked into the model weights.",
    },
    "scale": {
        "diagram_type": "scale",
        "title": "The Temperature Dial in LLMs",
        "subtitle": "One number that controls creativity vs safety",
        "axis_label": "Temperature (0 → 2)",
        "left_pole":  "Deterministic",
        "right_pole": "Chaotic",
        "markers": [
            {"position": 0.05, "label": "Temp 0.0", "detail": "Same answer every time"},
            {"position": 0.45, "label": "Temp 0.7", "detail": "Balanced — default range"},
            {"position": 0.90, "label": "Temp 1.8", "detail": "Random, often broken"},
        ],
        "takeaway": "Temperature doesn't make the model smarter — it just makes it take bigger guesses.",
    },
    "stats": {
        "diagram_type": "stats",
        "title": "LLM Scale — The Numbers That Matter",
        "subtitle": "What the parameter counts actually mean",
        "stats": [
            {"value": "1T+",  "label": "Parameters",    "detail": "Weights in the largest models"},
            {"value": "~4KB", "label": "Per Token",     "detail": "Memory for one KV cache entry"},
            {"value": "96",   "label": "Layers",        "detail": "Transformer blocks in GPT-4 class"},
            {"value": "128K", "label": "Context Window","detail": "Tokens held in memory at once"},
        ],
        "takeaway": "Bigger models aren't magic — they're just more matrix multiplications stacked up.",
    },
}


if __name__ == "__main__":
    import sys, subprocess
    dtype = sys.argv[1] if len(sys.argv) > 1 else "flow"
    out = f"renderer/output/test_{dtype}.png"

    if dtype == "all":
        for dt, sample in _SAMPLES.items():
            o = f"renderer/output/test_{dt}.png"
            render(sample, o)
            print(f"  saved {o}")
        subprocess.run(["open", "renderer/output/"])
    else:
        sample = _SAMPLES.get(dtype, _SAMPLES["flow"])
        render(sample, out)
        print(f"Saved → {out}")
        subprocess.run(["open", out])
