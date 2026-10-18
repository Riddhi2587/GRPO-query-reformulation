"""Training curves for the README: gain in R_cap@100 over the raw query, and completion length, per step.

    python -m eval.plot_training --run q1.7b_rcap100_lr2e-5_G8_r1_s1_2ep

Reads results/train/<run>/reward_metrics.jsonl and log_history.json; writes diagrams/training_curve.png.
Each step uses different queries, so the raw reward mostly shows how hard that step's queries were. The gain over the
raw query is measured on the same queries, which cancels that. Each step's value is the mean over its 64 samples, so the
faint line is noisy; the bold line is a trailing average.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# Surface and ink tokens, and categorical slots 1 and 2 of the dataviz reference palette (light mode).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last `window` values (fewer at the start), same length as `x`."""
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    n = np.arange(1, len(x) + 1)
    lo = np.maximum(n - window, 0)
    return (c[n] - c[lo]) / (n - lo)


def load_curves(run_dir: Path) -> dict[str, np.ndarray]:
    """Per-step R_cap@100 gain over the raw query, and mean completion length (tokens, incl. the <query> tags)."""
    rows = [json.loads(line) for line in (run_dir / "reward_metrics.jsonl").read_text().splitlines() if line]
    logs = [d for d in json.loads((run_dir / "log_history.json").read_text()) if "loss" in d]
    gain = np.array([r["rcap100_gain_over_raw"] for r in rows])
    length = np.array([d["completions/mean_length"] for d in logs])
    if len(gain) != len(length):
        raise ValueError(f"{len(gain)} reward rows but {len(length)} trainer log rows in {run_dir}")
    return dict(step=np.arange(1, len(gain) + 1), gain=gain, length=length)


def draw(curves: dict[str, np.ndarray], out: Path, window: int = 25, epoch_step: int | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, facecolor=SURFACE)
    panels = [("gain", "Gain in R_cap@100 over the raw query (same queries each step)", BLUE, "{:+.2f}"),
              ("length", "Completion length (tokens, includes the <query> tags)", ORANGE, "{:.1f}")]
    for ax, (key, title, color, fmt) in zip(axes, panels):
        y, smooth = curves[key], trailing_mean(curves[key], window)
        x = curves["step"]
        ax.set_facecolor(SURFACE)
        ax.plot(x, y, color=color, alpha=0.25, linewidth=1)
        ax.plot(x, smooth, color=color, linewidth=2, solid_capstyle="round")
        ax.plot(x[-1], smooth[-1], "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(fmt.format(smooth[-1]), (x[-1], smooth[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10)
        if epoch_step:
            ax.axvline(epoch_step, color=INK_2, linewidth=1, alpha=0.5)
        if key == "gain":  # the raw query is the baseline: zero gain
            ax.axhline(0, color=INK_2, linewidth=1, alpha=0.7)
            ax.annotate("raw query", (x[0], 0), xytext=(4, 4), textcoords="offset points", color=INK_2, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11, pad=8)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
    if epoch_step:
        axes[0].annotate("end of epoch 1", (epoch_step, 1.0), xycoords=("data", "axes fraction"), xytext=(-6, -4),
                         textcoords="offset points", ha="right", va="top", color=INK_2, fontsize=9)
    axes[1].set_xlabel("Training step", color=INK_2, fontsize=10)
    axes[1].set_xlim(1, curves["step"][-1] * 1.06)
    fig.text(0.01, 0.005, f"Faint line: each step (64 samples). Bold line: {window}-step average.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run name under results/train/")
    ap.add_argument("--out", default="diagrams/training_curve.png")
    ap.add_argument("--window", type=int, default=25, help="trailing-average window, in steps")
    ap.add_argument("--epoch-step", type=int, default=324, help="step to mark as the end of epoch 1 (0 for none)")
    a = ap.parse_args(argv)
    curves = load_curves(Path("results/train") / a.run)
    draw(curves, Path(a.out), a.window, a.epoch_step or None)
    print(f"wrote {a.out} ({len(curves['step'])} steps)")


if __name__ == "__main__":
    main()
