"""Metrics from a run log (one row per transaction, written by simulation.run_year).

    monthly(log)            one row per month: accuracy, manual checks, flips, memory size, ...
    summary(log)            the same numbers for the whole replay (or a span of months)
    line_chart(...)         one line per condition over the months, saved as a PNG

Definitions (EXPERIMENTS.md, "Metrics"):
    accuracy        shown label == true_category, before the user reviews it
    manual checks   rows the user had to act on: flagged ("Check this") + corrected
    flips           rows the frozen model had right that the shown label got wrong
    fixes           rows the frozen model had wrong that the shown label got right

Stability and plasticity, from the labels alone (works on any dataset, no event list needed).
For each row, look at earlier months' rows with the same text:
    adaptation row    its true label differs from the text's FIRST label (it changed, like coffee
                      after July), and that new label already appeared in an earlier month, so
                      memory had a chance to learn it
    adaptation error  an adaptation row labeled wrong (memory stuck on the old label)
    spread row        its true label is the text's first label, but the text had some other label
                      in an earlier month (Trader Joe's after the gift card)
    spread error      a spread row labeled with one of those other, earlier labels (the exception
                      leaked into the normal case)
    long-gap row      its text was seen before, but not in the last LONG_GAP months (rare items,
                      like a movie every few months); long-gap accuracy shows whether memory still
                      remembers them, which matters once memory has a size limit (Experiment 3)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LONG_GAP = 3   # months


def _add_flags(log: pd.DataFrame) -> pd.DataFrame:
    log = log.copy()
    log["correct"] = log["label"] == log["true_label"]
    log["frozen_correct"] = log["frozen_label"] == log["true_label"]
    log["corrected"] = log["source"] == "corrected"
    log["manual"] = log["flagged"] | log["corrected"]
    log["flip"] = log["frozen_correct"] & ~log["correct"]
    log["fix"] = ~log["frozen_correct"] & log["correct"]
    log["from_memory"] = log["path"] == "memory"
    log["adapt_row"], log["spread_row"], log["spread_error"], log["long_gap"] = _history_flags(log)
    log["adapt_error"] = log["adapt_row"] & ~log["correct"]
    return log


def _history_flags(log: pd.DataFrame):
    """adaptation row, spread row, spread error and long-gap row for each row (module docstring)."""
    first: dict[str, str] = {}
    last_month: dict[str, int] = {}            # last earlier month each text appeared in
    earlier: dict[str, set[str]] = {}          # labels each text had in earlier months
    this_month: dict[str, set[str]] = {}
    adapt, spread, spread_err, gap = [], [], [], []
    current = None
    for month, text, true, shown in zip(log["month"], log["text"], log["true_label"], log["label"]):
        if month != current:                   # a new month: last month's labels become "earlier"
            for t, labels in this_month.items():
                earlier.setdefault(t, set()).update(labels)
                last_month[t] = current
            this_month, current = {}, month
        first.setdefault(text, true)
        seen = earlier.get(text, set())
        others = seen - {true}
        is_adapt = true != first[text] and true in seen
        is_spread = true == first[text] and bool(others)
        adapt.append(is_adapt)
        spread.append(is_spread)
        spread_err.append(is_spread and shown in others)
        gap.append(text in last_month and month - last_month[text] >= LONG_GAP)
        this_month.setdefault(text, set()).add(true)
    return adapt, spread, spread_err, gap


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
        "adaptation_errors": int(g["adapt_error"].sum()),
        "adaptation_rows": int(g["adapt_row"].sum()),
        "spread_errors": int(g["spread_error"].sum()),
        "spread_rows": int(g["spread_row"].sum()),
        "long_gap_rows": int(g["long_gap"].sum()),
        "long_gap_accuracy": round(100 * g.loc[g["long_gap"], "correct"].mean(), 1) if g["long_gap"].any() else float("nan"),
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


def heatmap(table: pd.DataFrame, title: str, path: Path, xlabel: str, ylabel: str, fmt: str = "{:.0f}"):
    """A grid of numbers (e.g. manual checks for each trust x freshness pair). One hue, light to
    dark = small to large, with every value written in its cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#9cc0ee", "#2a78d6", "#123f7a"])
    fig, ax = plt.subplots(figsize=(1.3 * len(table.columns) + 2.4, 0.7 * len(table) + 1.8), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    values = table.to_numpy(dtype=float)
    ax.imshow(values, cmap=cmap, aspect="auto")
    lo, hi = values.min(), values.max()
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            dark = (values[r, c] - lo) / ((hi - lo) or 1) > 0.55
            ax.text(c, r, fmt.format(values[r, c]), ha="center", va="center", fontsize=10,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(table.columns)), [str(c) for c in table.columns], color=INK_2, fontsize=9)
    ax.set_yticks(range(len(table.index)), [str(i) for i in table.index], color=INK_2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


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
    top = ylim[1] if ylim else table.max().max()
    span = (ylim[1] - ylim[0]) if ylim else (top - table.min().min() or 1)
    gap = 0.055 * span
    ends.sort()
    for k in range(1, len(ends)):                       # push apart upwards...
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    if ends and ends[-1][0] > top:                       # ...then back down if past the top
        ends[-1][0] = top
        for k in range(len(ends) - 2, -1, -1):
            ends[k][0] = min(ends[k][0], ends[k + 1][0] - gap)
    for y, name, color in ends:
        ax.annotate(name, (table.index[-1], y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2, annotation_clip=False)
    raised = False
    prev = None
    for month, text in sorted((events or {}).items()):
        raised = (not raised) if prev is not None and month - prev <= 1 else False   # stagger neighbours
        ax.axvline(month - 0.5, color=GRID, linewidth=1, zorder=0)
        ax.text(month - 0.45, 1.07 if raised else 1.0, text, transform=ax.get_xaxis_transform(),
                fontsize=8, color=INK_2, va="bottom", ha="left")
        prev = month
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
    ncol = len(table.columns) if len(table.columns) <= 4 else 3
    rows = -(-len(table.columns) // ncol)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=ncol,
              frameon=False, fontsize=9, labelcolor=INK_2)
    fig.subplots_adjust(right=0.80, bottom=0.12 + 0.06 * rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
