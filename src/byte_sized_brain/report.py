"""Aggregate ``benchmarks/results/*.csv`` into a chart dashboard + ``docs/report.md``
and inject a results table + chart into the README.

Computes, per (pipeline, arch, emulated), the quantization trade-off relative to
that pipeline's FP32 baseline, meaning the size reduction, latency speedup and
accuracy delta. Emulated-ARM rows are kept separate from native rows so the story
stays honest. Only the top-level ``*.csv`` files are read, so smoke results (kept
in ``benchmarks/results/smoke/``) never leak into the report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import results_root
from .utils import get_logger, normalize_arch

log = get_logger("report")

_DOCS = Path("docs")
_IMAGES = _DOCS / "images"
_README = Path("README.md")
_README_BEGIN = "<!-- RESULTS_TABLE -->"
_README_END = "<!-- /RESULTS_TABLE -->"
_DASHBOARD = _IMAGES / "results_dashboard.png"

# Friendly, compact axis labels (what the model reads, then what it is).
_LABELS = {
    "ffn_mnist": "Digits\nFFN",
    "cnn_cifar10": "Photos\nCNN",
    "rnn_imdb": "Reviews\nLSTM",
    "distilbert_imdb": "Reviews\nDistilBERT",
}
_ORDER = ["ffn_mnist", "cnn_cifar10", "rnn_imdb", "distilbert_imdb"]

# Plain-language names for the README table. The CSVs keep the identifiers.
_FRIENDLY = {
    "ffn_mnist": "Digit reader (MNIST)",
    "cnn_cifar10": "Photo classifier (CIFAR-10)",
    "rnn_imdb": "Review sentiment, LSTM (IMDB)",
    "distilbert_imdb": "Review sentiment, DistilBERT (IMDB)",
}
# TFLite's "dynamic-range" and ONNX Runtime's "dynamic" quantization are the same
# idea (INT8 weights, activation ranges measured on the fly), so the README uses
# one name for both.
_METHOD = {
    "static_int8": "Static INT8",
    "dynamic_range": "Dynamic INT8",
    "dynamic_int8": "Dynamic INT8",
}


def load_results(results_dir: Path | None = None) -> pd.DataFrame:
    results_dir = results_dir or results_root()
    csvs = sorted(p for p in results_dir.glob("*.csv"))
    if not csvs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    # Older rows say AMD64 (Windows) where newer ones say x86_64 (Linux); same ISA.
    df["arch"] = df["arch"].map(normalize_arch)
    df["emulated"] = df["emulated"].astype(str).str.strip().str.lower().isin({"true", "1"})
    return df


def _baseline_map(df: pd.DataFrame) -> dict[tuple, pd.Series]:
    base = df[df["variant"] == "fp32"]
    return {(r["pipeline"], r["arch"], r["emulated"]): r for _, r in base.iterrows()}


def wilson_halfwidth(accuracy: float, n: int, z: float = 1.96) -> float:
    """Half-width of the 95 percent Wilson score interval for an accuracy."""
    if n <= 0:
        return float("nan")
    denom = 1 + z * z / n
    spread = z * ((accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) ** 0.5)
    return spread / denom


def headline_arch(df: pd.DataFrame) -> str:
    """The architecture the README table shows: the one with the most rows, x86_64 on a tie."""
    counts = df["arch"].value_counts()
    top = sorted(counts[counts == counts.max()].index)
    return "x86_64" if "x86_64" in top else top[0]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    baselines = _baseline_map(df)
    rows = []
    for _, r in df.iterrows():
        b = baselines.get((r["pipeline"], r["arch"], r["emulated"]))
        size_red = lat_speedup = acc_delta = float("nan")
        if b is not None and b["size_mb"]:
            size_red = 100.0 * (b["size_mb"] - r["size_mb"]) / b["size_mb"]
            if r["latency_ms_mean"]:
                lat_speedup = b["latency_ms_mean"] / r["latency_ms_mean"]
            acc_delta = r["accuracy"] - b["accuracy"]
        rows.append(
            {
                "pipeline": r["pipeline"],
                "modality": r["modality"],
                "runtime": r["runtime"],
                "variant": r["variant"],
                "arch": r["arch"],
                "emulated": bool(r["emulated"]),
                "size_mb": round(r["size_mb"], 3),
                "accuracy": round(r["accuracy"], 4),
                "accuracy_ci95": "±"
                + format(100 * wilson_halfwidth(r["accuracy"], int(r["num_samples"])), ".1f")
                + " pts"
                if "num_samples" in r
                else "",
                "agreement": round(r["agreement"], 4)
                if "agreement" in r and pd.notna(r["agreement"])
                else "",
                "latency_ms_mean": round(r["latency_ms_mean"], 3),
                "size_reduction_%": round(size_red, 1),
                "latency_speedup_x": round(lat_speedup, 2),
                "accuracy_delta": round(acc_delta, 4),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _fmt_mb(x: float) -> str:
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def _fmt_ms(x: float) -> str:
    if x < 0.1:
        return f"{x:.3f}"
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def _speed_words(fp32_ms: float, q_ms: float) -> str:
    """'1.7× faster', '1.5× slower' or 'about the same'.

    Anything within 15 percent counts as the same, because single-run timings on a
    laptop move by that much from one run to the next.
    """
    if not fp32_ms or not q_ms:
        return "n/a"
    ratio = fp32_ms / q_ms
    if 0.85 <= ratio <= 1.15:
        return "about the same"
    return f"{ratio:.1f}× faster" if ratio > 1 else f"{1 / ratio:.1f}× slower"


def _readme_table(df: pd.DataFrame, arch: str) -> str:
    """One plain-language row per model: original (FP32) versus quantized, native only.

    The identifier-heavy per-variant table stays in docs/report.md for specialists.
    """
    native = df[(df["arch"] == arch) & (~df["emulated"].astype(bool))]
    present = set(native["pipeline"])
    order = [p for p in _ORDER if p in present] + sorted(present - set(_ORDER))
    header = [
        "Model (dataset)",
        "Method",
        "Size on disk",
        "Accuracy",
        "Time per input",
        "Memory",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for p in order:
        sub = native[native["pipeline"] == p]
        fp, q = sub[sub["variant"] == "fp32"], sub[sub["variant"] != "fp32"]
        if fp.empty or q.empty:
            continue
        fp, q = fp.iloc[0], q.iloc[0]
        smaller = 100.0 * (fp["size_mb"] - q["size_mb"]) / fp["size_mb"] if fp["size_mb"] else 0.0
        pts = 100.0 * (q["accuracy"] - fp["accuracy"])
        change = "no change" if abs(pts) < 0.05 else f"{pts:+.1f} pts"
        cells: list[str] = [
            _FRIENDLY.get(str(p), str(p)),
            _METHOD.get(str(q.get("quantization", "")), str(q["variant"])),
            f"{_fmt_mb(fp['size_mb'])} → {_fmt_mb(q['size_mb'])} MB ({smaller:.0f}% smaller)",
            f"{100 * fp['accuracy']:.1f}% → {100 * q['accuracy']:.1f}% ({change})",
            f"{_fmt_ms(fp['latency_ms_mean'])} → {_fmt_ms(q['latency_ms_mean'])} ms "
            f"({_speed_words(fp['latency_ms_mean'], q['latency_ms_mean'])})",
            f"{_fmt_mb(fp['rss_delta_mb'])} → {_fmt_mb(q['rss_delta_mb'])} MB",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def inject_readme_table(
    table_md: str, arch: str, chart_rel: str | None = None, readme: Path = _README
) -> bool:
    """Replace the span between the RESULTS_TABLE sentinels with chart + table."""
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if _README_BEGIN not in text or _README_END not in text:
        return False
    pre, rest = text.split(_README_BEGIN, 1)
    _, post = rest.split(_README_END, 1)
    chart = f"\n![Quantization trade-offs]({chart_rel})\n" if chart_rel else ""
    block = (
        f"{_README_BEGIN}\n\n{table_md}\n\n_Each cell reads original (FP32) → quantized "
        f"(INT8). Time is the average for one input on a native {arch} CPU, and memory is "
        f"what the process gains by loading and running the model, so lower is better for "
        f"both. Generated by `bsb report` from `benchmarks/results/*.csv`, which also "
        f"record p50 and p95 latency. The per-variant table is in "
        f"[docs/report.md](docs/report.md)._\n{chart}\n{_README_END}"
    )
    readme.write_text(pre + block + post, encoding="utf-8", newline="\n")
    return True


def _paired(df: pd.DataFrame, arch: str) -> pd.DataFrame:
    """One row per pipeline pairing the FP32 baseline with its quantized variant."""
    d = df[df["arch"] == arch]
    present = set(d["pipeline"])
    rows = []
    for p in [x for x in _ORDER if x in present] + [x for x in present if x not in _ORDER]:
        sub = d[d["pipeline"] == p]
        fp = sub[sub["variant"] == "fp32"]
        q = sub[sub["variant"] != "fp32"]
        if fp.empty or q.empty:
            continue
        fp, q = fp.iloc[0], q.iloc[0]
        rows.append(
            {
                "label": _LABELS.get(p, p),
                "fp32_size": fp["size_mb"],
                "q_size": q["size_mb"],
                "fp32_acc": fp["accuracy"],
                "q_acc": q["accuracy"],
                "fp32_lat": fp["latency_ms_mean"],
                "q_lat": q["latency_ms_mean"],
                "reduction": 100.0 * (fp["size_mb"] - q["size_mb"]) / fp["size_mb"]
                if fp["size_mb"]
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _dashboard(paired: pd.DataFrame, arch: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if paired.empty:
        raise ValueError("no paired fp32/quantized rows to chart")

    labels = list(paired["label"])
    x = np.arange(len(labels))
    w = 0.38
    fp_color, q_color = "#9aa7b8", "#2e7d32"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"What 8-bit quantization changed, model by model ({arch} CPU)",
        fontsize=14,
        fontweight="bold",
    )

    def grouped(ax, fp_vals, q_vals, title, ylabel, *, log=False, fmt="%.2f"):
        b1 = ax.bar(x - w / 2, fp_vals, w, label="original (FP32)", color=fp_color)
        b2 = ax.bar(x + w / 2, q_vals, w, label="quantized (INT8)", color=q_color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        if log:
            ax.set_yscale("log")
        ax.bar_label(b1, fmt=fmt, fontsize=7, padding=2)
        ax.bar_label(b2, fmt=fmt, fontsize=7, padding=2)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    grouped(
        axes[0, 0],
        paired["fp32_size"],
        paired["q_size"],
        "Size on disk (lower is better)",
        "MB, log scale",
        log=True,
    )
    grouped(
        axes[0, 1],
        100 * paired["fp32_acc"],
        100 * paired["q_acc"],
        "Accuracy on held-out test data (higher is better)",
        "% correct",
        fmt="%.1f%%",
    )
    axes[0, 1].set_ylim(0, 108)
    grouped(
        axes[1, 0],
        paired["fp32_lat"],
        paired["q_lat"],
        "Time per input (lower is better)",
        "ms, log scale",
        log=True,
    )

    bars = axes[1, 1].bar(x, paired["reduction"], w * 1.6, color="#1565c0")
    axes[1, 1].set_title("How much smaller the quantized model is")
    axes[1, 1].set_ylabel("% smaller")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, fontsize=8)
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def generate_report(results_dir: Path | None = None) -> Path | None:
    df = load_results(results_dir)
    if df.empty:
        log.warning("No results in %s. Run `bsb benchmark <pipeline>` first.", results_root())
        return None

    summary = summarize(df)
    native = df[~df["emulated"].astype(bool)]
    headline = native if not native.empty else df
    # Most-benchmarked native architecture. A tie goes to x86_64, so adding ARM rows
    # for the same pipelines never silently flips the README headline.
    primary_arch = headline_arch(headline)

    chart_ok = False
    try:
        _dashboard(_paired(headline, primary_arch), primary_arch, _DASHBOARD)
        chart_ok = True
    except Exception as exc:  # never let a plotting hiccup kill the report
        log.warning("Could not render dashboard: %s", exc)

    report_path = _DOCS / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Results\n",
        "_Generated by `bsb report` from `benchmarks/results/*.csv`._\n",
    ]
    if chart_ok:
        parts += [
            f"## Quantization trade-offs ({primary_arch})\n",
            f"![Quantization trade-offs]({_DASHBOARD.relative_to(_DOCS).as_posix()})\n",
        ]
    parts += [
        "## Per-variant detail\n",
        _markdown_table(summary),
        "",
        "## Notes\n",
        "- `size_reduction_%` is how much smaller the variant is than its FP32 original. "
        "`latency_speedup_x` is FP32 time divided by variant time, so above 1 means "
        "faster and below 1 means slower. `accuracy_delta` is the change in accuracy as "
        "a fraction (-0.173 means 17.3 percentage points lower).",
        "- `accuracy_ci95` is the half-width of the 95 percent Wilson interval for the "
        "accuracy at that sample size, so a difference smaller than it is within noise. "
        "`agreement` is the share of test examples where the variant gives the same "
        "answer as its FP32 baseline.",
        "- All three are computed against each pipeline's own FP32 baseline, within the "
        "same architecture. Sizes are in MiB (1,048,576 bytes) and latency is the mean "
        "time for one input, in milliseconds.",
        "- Rows with `emulated = True` come from QEMU-emulated ARM64. Their **absolute "
        "latency is not comparable** to native runs, so treat it as a relative ratio only "
        "(see the [methodology](methodology.md)).",
    ]
    report_path.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    log.info("Wrote %s (dashboard: %s)", report_path, chart_ok)

    chart_rel = _DASHBOARD.as_posix() if chart_ok else None
    if inject_readme_table(_readme_table(headline, primary_arch), primary_arch, chart_rel):
        log.info("Injected results table + chart into %s", _README)

    return report_path
