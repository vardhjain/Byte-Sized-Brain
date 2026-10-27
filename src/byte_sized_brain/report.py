"""Aggregate ``benchmarks/results/*.csv`` into a chart dashboard + ``docs/report.md``
and inject a results table + chart into the README.

Computes trade-offs against the FP32 baseline from the *same benchmark run*,
using run_id to avoid mixing results from different devices or experiments. Emulated-ARM rows are kept separate from native rows so the story
stays honest. Only the top-level ``*.csv`` files are read, so smoke results (kept
in ``benchmarks/results/smoke/``) never leak into the report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .benchmark.metrics import effective_run_id
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


def _with_run_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Make both new and historical result frames safe to pair by run ID."""
    if df.empty:
        return df
    if (
        "run_id" not in df.columns
        or df["run_id"].isna().any()
        or (df["run_id"].astype(str).str.strip() == "").any()
    ):
        df = df.copy()
        df["run_id"] = df.apply(lambda row: effective_run_id(row.to_dict()), axis=1)
    return df


def load_results(results_dir: Path | None = None) -> pd.DataFrame:
    results_dir = results_dir or results_root()
    csvs = sorted(p for p in results_dir.glob("*.csv"))
    if not csvs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    # Older rows say AMD64 (Windows) where newer ones say x86_64 (Linux); same ISA.
    df["arch"] = df["arch"].map(normalize_arch)
    df["emulated"] = df["emulated"].astype(str).str.strip().str.lower().isin({"true", "1"})
    return _with_run_ids(df)


def _baseline_map(df: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    base = df[df["variant"] == "fp32"]
    return {(str(r["pipeline"]), str(r["run_id"])): r for _, r in base.iterrows()}


def _featured_pairs(df: pd.DataFrame, arch: str) -> list[tuple[str, pd.Series, pd.Series]]:
    """Choose the newest complete, native benchmark run for each pipeline.

    A newer FP32-only or INT8-only run is ignored rather than being paired with
    another experiment. Sorting by the recorded timestamps keeps selection
    deterministic regardless of CSV insertion order.
    """
    native = _with_run_ids(df)
    native = native[(native["arch"] == arch) & (~native["emulated"].astype(bool))]
    present = set(native["pipeline"])
    order = [p for p in _ORDER if p in present] + sorted(present - set(_ORDER))
    pairs: list[tuple[str, pd.Series, pd.Series]] = []
    for p in order:
        candidates = []
        for run_id, sub in native[native["pipeline"] == p].groupby("run_id", sort=False):
            fp = sub[sub["variant"] == "fp32"]
            # Preserve the headline's published static INT8 comparison even
            # when a run also includes dynamic-range and selective variants.
            preferred = "int8" if p == "cnn_cifar10" else None
            q = sub[sub["variant"] == preferred] if preferred else sub[sub["variant"] != "fp32"]
            if fp.empty or q.empty:
                continue
            latest = max(sub["timestamp"].astype(str)) if "timestamp" in sub else ""
            candidates.append((latest, str(run_id), fp.iloc[-1], q.iloc[-1]))
        if candidates:
            _, _, fp, q = max(candidates, key=lambda candidate: candidate[:2])
            pairs.append((str(p), fp, q))
    return pairs


def wilson_halfwidth(accuracy: float, n: int, z: float = 1.96) -> float:
    """Half-width of the 95 percent Wilson score interval for an accuracy."""
    if n <= 0:
        return float("nan")
    denom = 1 + z * z / n
    spread = z * ((accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) ** 0.5)
    return spread / denom


def headline_arch(df: pd.DataFrame) -> str:
    """Featured architecture: most distinct pipelines, x86_64 on a tie."""
    # Repeating the same machine must not bias the featured architecture.
    counts = df.groupby("arch")["pipeline"].nunique()
    top = sorted(counts[counts == counts.max()].index)
    return "x86_64" if "x86_64" in top else top[0]


def _optional_number(row: pd.Series, name: str, *, decimals: int = 4) -> float | str:
    value = row.get(name)
    return round(float(value), decimals) if pd.notna(value) and value != "" else ""


def _paired_delta_interval(row: pd.Series) -> str:
    low = _optional_number(row, "accuracy_delta_ci95_low")
    high = _optional_number(row, "accuracy_delta_ci95_high")
    if low == "" or high == "":
        return ""
    return f"[{float(low) * 100:+.1f}, {float(high) * 100:+.1f}] pts"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _with_run_ids(df)
    baselines = _baseline_map(df)
    rows = []
    for _, r in df.iterrows():
        b = baselines.get((str(r["pipeline"]), str(r["run_id"])))
        size_red = lat_speedup = acc_delta = float("nan")
        if b is not None and b["size_mb"]:
            size_red = 100.0 * (b["size_mb"] - r["size_mb"]) / b["size_mb"]
            if r["latency_ms_mean"]:
                lat_speedup = b["latency_ms_mean"] / r["latency_ms_mean"]
            acc_delta = r["accuracy"] - b["accuracy"]
        rows.append(
            {
                "pipeline": r["pipeline"],
                "run_id": r["run_id"],
                "device": r.get("device", ""),
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
                "latency_passes": _optional_number(r, "latency_passes", decimals=0),
                "latency_pass_std_ms": _optional_number(r, "latency_pass_std_ms"),
                "latency_ms_ci95_low": _optional_number(r, "latency_ms_ci95_low"),
                "latency_ms_ci95_high": _optional_number(r, "latency_ms_ci95_high"),
                "paired_accuracy_delta_ci95": _paired_delta_interval(r),
                "mcnemar_pvalue": float(r["mcnemar_pvalue"])
                if pd.notna(r.get("mcnemar_pvalue")) and r.get("mcnemar_pvalue") != ""
                else "",
                "fp32_only_correct": _optional_number(r, "fp32_only_correct", decimals=0),
                "variant_only_correct": _optional_number(r, "variant_only_correct", decimals=0),
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
    header = [
        "Model (dataset)",
        "Method",
        "Size on disk",
        "Accuracy",
        "Time per input",
        "Memory",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for p, fp, q in _featured_pairs(df, arch):
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
    rows = []
    for p, fp, q in _featured_pairs(df, arch):
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
    # Feature the architecture with the most distinct pipelines, breaking ties
    # toward x86_64. Additional reruns must not flip the README headline.
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
        "- `accuracy_ci95` is the per-model 95 percent Wilson half-width; **do not "
        "use it to test differences between paired models**. "
        "`paired_accuracy_delta_ci95` is a seeded bootstrap interval in percentage "
        "points using the same held-out examples for FP32 and the variant. "
        "`mcnemar_pvalue` is McNemar's exact two-sided test on correctness changes. "
        "`fp32_only_correct` / `variant_only_correct` count the discordant examples. "
        "`agreement` is the share of examples with the same predicted label.",
        "- `latency_passes` counts complete, independently repeated timing passes. "
        "`latency_ms_ci95_low/high` bootstrap the mean latency per pass; "
        "`latency_pass_std_ms` is the sample standard deviation of those means. "
        "These are blank when fewer than two passes were collected, or for legacy CSVs. "
        "Pass-to-pass variation describes this machine and benchmark session only.",
        "- All three are computed against the FP32 baseline within the exact same run_id, "
        "never just the same architecture. Sizes are in MiB (1,048,576 bytes) and latency is the mean "
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
