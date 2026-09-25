import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image


matplotlib.use("Agg")
import matplotlib.pyplot as plt


base = Path(__file__).parent
job = json.loads((base / "job.json").read_text())
current = Path(job["local_output_directory"])
if current.is_absolute():
    # Archived jobs recorded the original author's checkout location.
    current = base / "runs" / current.parent.name / current.name
else:
    current = base / current
previous = base.parent / "pretrained/results"
comparison = current.parent / "comparison"
comparison.mkdir(exist_ok=True)
rows = []
for count, root in ((1, previous), (3, current)):
    metadata = json.loads((root / "metadata.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    for row in summary["measurements"]:
        rows.append({"reference_images": count, **row})
    with (root / "runs.csv").open() as handle:
        runs = list(csv.DictReader(handle))
    assert len(runs) == 32
    for run in runs:
        image = Image.open(root / run["image"])
        assert image.size == (1024, 1024)
        assert hashlib.sha256(image.tobytes()).hexdigest() == run["pixel_sha256"]
    for steps in ("4", "8"):
        selected = [run for run in runs if run["steps"] == steps]
        print(count, "references", steps, "steps", "unique pixel hashes:", len({r["pixel_sha256"] for r in selected}))
    if count == 1:
        previous_metadata = metadata
    else:
        for key in (
            "model",
            "revision",
            "prompt",
            "seed",
            "output_size",
            "dtype",
            "gpu",
            "torch",
            "cuda",
            "attention_backend",
        ):
            assert metadata[key] == previous_metadata[key], key
        for name, version in previous_metadata["packages"].items():
            assert metadata["packages"][name] == version, name
        assert metadata["reference_count"] == 3
        assert all(ref["size"] == [1024, 704] for ref in metadata["references"])
        assert metadata["references"][0]["sha256"] == previous_metadata["reference_sha256"]
with (comparison / "comparison.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
effects = []
for count in (1, 3):
    for steps in (4, 8):
        uncached, cached = [
            next(r for r in rows if r["reference_images"] == count and r["steps"] == steps and r["mode"] == mode)
            for mode in ("uncached", "cached")
        ]
        effects.append(
            {
                "reference_images": count,
                "steps": steps,
                "speedup": uncached["median_seconds"] / cached["median_seconds"],
                "latency_reduction_percent": 100 * (1 - cached["median_seconds"] / uncached["median_seconds"]),
                "extra_peak_memory_gib": cached["max_peak_allocated_gib"] - uncached["max_peak_allocated_gib"],
            }
        )
(comparison / "effects.json").write_text(json.dumps(effects, indent=2) + "\n")
print(json.dumps(effects, indent=2))
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
fig.subplots_adjust(left=0.075, right=0.985, top=0.72, bottom=0.23, wspace=0.25)
fig.suptitle("KV caching: one vs. three reference images", x=0.075, y=0.97, ha="left", fontsize=21, fontweight="bold")
fig.text(
    0.075,
    0.855,
    "FLUX.2 Klein 9B KV · A100 80 GB · bfloat16 · 1024 × 1024 output\n3 warmups + 5 measured runs per setting · same prompt and seed",
    fontsize=11,
    color="#475569",
)
settings = [(4, 1), (4, 3), (8, 1), (8, 3)]
x = np.arange(len(settings))
for mode, shift, color, label in (
    ("uncached", -0.18, "#94a3b8", "Without cache"),
    ("cached", 0.18, "#2563eb", "With cache"),
):
    selected = [
        next(r for r in rows if r["steps"] == steps and r["reference_images"] == count and r["mode"] == mode)
        for steps, count in settings
    ]
    for ax, metric in zip(axes, ("median_seconds", "max_peak_allocated_gib")):
        values = [r[metric] for r in selected]
        bars = ax.bar(x + shift, values, width=0.34, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
        if metric == "median_seconds":
            ax.errorbar(
                x + shift,
                values,
                yerr=[
                    [r[metric] - r["min_seconds"] for r in selected],
                    [r["max_seconds"] - r[metric] for r in selected],
                ],
                fmt="none",
                ecolor="#334155",
                capsize=3,
            )
for ax, title, ylabel in zip(
    axes, ("Full-pipeline latency", "Peak GPU memory"), ("Seconds (median)", "Allocated memory (GiB)")
):
    ax.set_title(title, loc="left", fontsize=14, pad=14)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, [f"{count} ref{'s' if count > 1 else ''}\n{steps} steps" for steps, count in settings])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=10)
    ax.grid(axis="y", color="#e2e8f0")
    ax.set_axisbelow(True)
fig.text(
    0.075,
    0.055,
    "Includes encoding and decoding; excludes PNG saving. Memory includes model weights. Whiskers show the observed timing range.",
    fontsize=9,
    color="#475569",
)
for extension in ("png", "svg", "pdf"):
    fig.savefig(comparison / f"reference_count_comparison.{extension}", dpi=200, facecolor="white")
plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(9, 9.8))
fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.025, hspace=0.1, wspace=0.03)
fig.suptitle("Three references: cached vs. recomputed", x=0.03, ha="left", fontsize=19, fontweight="bold")
fig.text(0.03, 0.93, "Seed 42 · first measured output in each setting", fontsize=11, color="#475569")
for row, steps in enumerate((4, 8)):
    for col, mode in enumerate(("uncached", "cached")):
        ax = axes[row, col]
        ax.imshow(Image.open(current / f"{steps}_steps/{mode}/measure_01_seed_42.png"))
        ax.set_title(
            f"{steps} steps · " + ("Without cache" if mode == "uncached" else "With cache"), loc="left", fontsize=12
        )
        ax.axis("off")
fig.savefig(comparison / "image_comparison.png", dpi=180, facecolor="white")
plt.close(fig)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
for index, ax in enumerate(axes, start=1):
    ax.imshow(Image.open(current / f"reference_{index:02d}.png"))
    ax.set_title(f"Reference {index}", loc="left")
    ax.axis("off")
fig.tight_layout()
fig.savefig(comparison / "references.png", dpi=180, facecolor="white")
plt.close(fig)
print(comparison)
