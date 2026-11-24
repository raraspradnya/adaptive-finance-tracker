"""Metrics from a run log (one row per transaction, written by simulation.run_year).

    monthly(log)            one row per month: accuracy, manual checks, flips, memory size, ...
    summary(log)            the same numbers for the whole replay (or a span of months)
    line_chart(...)         one line per condition over the months, saved as a PNG

Definitions (EXPERIMENTS.md, "Metrics"):
    accuracy        shown label == true_category, before the user reviews it
    manual checks   rows the user had to act on: flagged ("Check this") + corrected
    flips           rows the frozen model had right that the shown label got wrong
    fixes           rows the frozen model had wrong that the shown label got right
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _add_flags(log: pd.DataFrame) -> pd.DataFrame:
    log = log.copy()
    log["correct"] = log["label"] == log["true_label"]
    log["frozen_correct"] = log["frozen_label"] == log["true_label"]
    log["corrected"] = log["source"] == "corrected"
    log["manual"] = log["flagged"] | log["corrected"]
    log["flip"] = log["frozen_correct"] & ~log["correct"]
    log["fix"] = ~log["frozen_correct"] & log["correct"]
    log["from_memory"] = log["path"] == "memory"
    return log


def _numbers(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "rows": len(g),
        "accuracy": round(100 * g["correct"].mean(), 1),
        "frozen_accuracy": round(100 * g["frozen_correct"].mean(), 1),
        "manual_checks": int(g["manual"].sum()),
        "flagged": int(g["flagged"].sum()),
        "corrected": int(g["corrected"].sum()),
        "from_memory": int(g["from_memory"].sum()),
        "flips": int(g["flip"].sum()),
        "fixes": int(g["fix"].sum()),
        "fix_rate": round(100 * g["fix"].sum() / max(1, (~g["frozen_correct"]).sum()), 1),
    })


def monthly(log: pd.DataFrame) -> pd.DataFrame:
    log = _add_flags(log)
    out = log.groupby("month").apply(_numbers, include_groups=False)
    out["memory_size"] = log.groupby("month")["memory_size"].last()
    return out


def summary(log: pd.DataFrame, months: range | None = None) -> pd.Series:
    log = _add_flags(log)
    if months is not None:
        log = log[log["month"].isin(months)]
    s = _numbers(log)
    s["memory_size"] = int(log["memory_size"].iloc[-1])
    return s


# ---------------------------------------------------------------- charts

# Categorical slots from the dataviz reference palette (light mode), in fixed order.
# The frozen-only baseline is drawn in neutral gray, dashed, so it reads as the reference line.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300"]
BASELINE = "#8a8984"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def line_chart(table: pd.DataFrame, title: str, ylabel: str, path: Path,
               baseline: str | None = None, events: dict[int, str] | None = None,
               ylim: tuple[float, float] | None = None):
    """table: index = month (1-12), one column per condition. Direct labels at the line ends."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    slot = 0
    ends = []
    for name in table.columns:
        y = table[name]
        if name == baseline:
            color, style = BASELINE, (0, (4, 3))
        else:
            color, style = SERIES[slot], "-"
            slot += 1
        ax.plot(table.index, y, color=color, linestyle=style, linewidth=2,
                marker="o", markersize=4.5, label=name, solid_capstyle="round")
        ends.append([float(y.iloc[-1]), name, color])
    # direct labels at the right end, nudged apart so they don't collide
    span = (ylim[1] - ylim[0]) if ylim else (table.max().max() - table.min().min() or 1)
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.055 * span)
    for y, name, color in ends:
        ax.annotate(name, (table.index[-1], y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    for month, text in (events or {}).items():
        ax.axvline(month - 0.5, color=GRID, linewidth=1, zorder=0)
        ax.text(month - 0.45, 1.0, text, transform=ax.get_xaxis_transform(), fontsize=8,
                color=INK_2, va="bottom", ha="left")
    ax.set_xticks(range(1, 13), MONTHS, color=INK_2, fontsize=9)
    ax.tick_params(axis="y", colors=INK_2, labelsize=9, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=22)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=len(table.columns),
              frameon=False, fontsize=9, labelcolor=INK_2)
    fig.subplots_adjust(right=0.80, bottom=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
