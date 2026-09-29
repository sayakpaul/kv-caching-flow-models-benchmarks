import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont


matplotlib.use("Agg")
import matplotlib.pyplot as plt


base = Path(__file__).parent
job = json.loads((base / "job.json").read_text())
results = Path(job["local_output_directory"])
if results.is_absolute():
    # Archived jobs recorded the original checkout location.
    results = base / "runs" / results.parent.name / results.name
else:
    results = base / results
run = results.parent
artifact_base = (
    "https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/"
    f"qwenimage21-benchmark/runs/{run.name}"
)
metadata = json.loads((results / "metadata.json").read_text())
summary = json.loads((results / "summary.json").read_text())
validation = json.loads((results / "validation.json").read_text())
rows = list(csv.DictReader((results / "runs.csv").open()))
modes = ("without_cache", "with_cache")
lookup = {(x["steps"], x["mode"]): x for x in summary}
assert metadata["steps"] == [40]
assert len(rows) == 16 and len(validation) == 2
for row in [*rows, *validation]:
    image = Image.open(results / row["image"])
    assert hashlib.sha256(image.tobytes()).hexdigest() == row["pixel_sha256"]
for item in validation:
    audit = item["audit"]
    assert len(audit) == 40 and all(x["causal_condition"] for x in audit)
    if item["mode"] == "with_cache":
        assert audit[0]["mode"] == "extract"
        assert all(x["mode"] == "cached" and x["output_tokens"] == 4096 for x in audit[1:])
        assert all(x["cache_layers"] == 32 for x in audit)
    else:
        assert all(x["mode"] is None and x["cache_layers"] == 0 for x in audit)
assert validation[0]["first_latent_sha256"] == validation[1]["first_latent_sha256"]
for item in summary:
    group = [x for x in rows if x["mode"] == item["mode"]]
    assert sum(x["phase"] == "warmup" for x in group) == 3
    assert sum(x["phase"] == "measure" for x in group) == 5
    expected = next(x["pixel_sha256"] for x in validation if x["mode"] == item["mode"])
    assert all(x["pixel_sha256"] == expected for x in group)
a, b = [lookup[(40, mode)] for mode in modes]
reduction = 100 * (1 - b["median_seconds"] / a["median_seconds"])
table = [
    "| Measurement | Without KV cache | With KV cache |",
    "| --- | ---: | ---: |",
    f"| Latency (40 steps) | {a['median_seconds']:.3f}s | {b['median_seconds']:.3f}s ({reduction:.1f}% lower) |",
    f"| Peak GPU memory | {a['max_peak_allocated_gib']:.2f} GiB | {b['max_peak_allocated_gib']:.2f} GiB |",
]
(run / "table.md").write_text("\n".join(table) + "\n")
(run / "changes.json").write_text(
    json.dumps(
        {
            "latency_reduction_percent": reduction,
            "speedup": a["median_seconds"] / b["median_seconds"],
            "peak_memory_change_gib": b["max_peak_allocated_gib"] - a["max_peak_allocated_gib"],
        },
        indent=2,
    )
)


def font(size, bold=False):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf", size)


canvas = Image.new("RGB", (1200, 748), "#F7F9FC")
draw = ImageDraw.Draw(canvas)
draw.text((48, 26), "Qwen-Image-2.1: KV caching", font=font(34, True), fill="#18283C")
draw.text((48, 80), "40 steps · seed 42 · same prompt and reference image", font=font(21), fill="#5C6D81")
for column, (mode, label) in enumerate(zip(modes, ("Without KV cache", "With KV cache"))):
    x = 48 + column * 576
    draw.text((x, 139), label, font=font(26, True), fill="#18283C")
    path = next(x["image"] for x in validation if x["mode"] == mode)
    sample = Image.open(results / path).convert("RGBA")
    background = Image.new("RGBA", sample.size, "white")
    sample = Image.alpha_composite(background, sample).convert("RGB")
    canvas.paste(sample.resize((528, 528), Image.Resampling.LANCZOS), (x, 188))
canvas.save(run / "comparison_seed_42.png")

plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
)
fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.26, top=0.70, wspace=0.38)
fig.text(0.1, 0.93, "Qwen-Image-2.1: KV cache on vs. off", fontsize=22, weight="bold")
fig.text(0.1, 0.86, "40 steps · A100 80 GB · bfloat16 · 1024 × 1024 · one reference", fontsize=11, color="#526072")
for ax, field, title, unit in zip(
    axes, ("median_seconds", "max_peak_allocated_gib"), ("End-to-end latency", "Peak GPU memory"), ("Seconds", "GiB")
):
    values = [x[field] for x in (a, b)]
    bars = ax.bar([0, 1], values, 0.58, color=["#3366CC", "#008B80"], zorder=3)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=6)
    if field == "median_seconds":
        errors = [[x[field] - x["min_seconds"] for x in (a, b)], [x["max_seconds"] - x[field] for x in (a, b)]]
        ax.errorbar([0, 1], values, yerr=errors, fmt="none", ecolor="#263443", capsize=3, linewidth=1, zorder=4)
    ax.set_xticks([0, 1], ("Without cache", "With cache"))
    ax.set_title(title, loc="left", fontsize=14, pad=14)
    ax.set_ylabel(unit)
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", color="#E3E8EE", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#CDD5DE")
fig.text(0.1, 0.13, "3 warmups + 5 timed runs · median latency; whiskers show min–max", fontsize=10, color="#526072")
fig.text(
    0.1,
    0.078,
    "Peak allocated memory includes model weights. Cache stores text + reference K/V.",
    fontsize=10,
    color="#526072",
)
for extension in ("png", "svg", "pdf"):
    fig.savefig(run / f"qwenimage21_performance.{extension}", dpi=220, facecolor="white")
plt.close(fig)
cached_audit = next(x["audit"][0] for x in validation if x["mode"] == "with_cache")
report = "\n".join(
    [
        "# Qwen-Image-2.1: with and without KV caching",
        "",
        *table,
        "",
        "A100-SXM4-80GB, bfloat16, batch size 1, 1024 × 1024 output, 40 denoising steps, one cat reference, seed 42, no classifier-free guidance. Each mode has three warmups followed by five timed calls, with alternating mode order. The table uses median end-to-end latency and the largest peak allocated GPU memory across the five measured calls.",
        "",
        f"![Performance]({artifact_base}/qwenimage21_performance.png)",
        "",
        "## What is compared",
        "",
        "The pipeline's built-in `use_kv_cache=False` and `use_kv_cache=True` settings. `causal_condition=True` remains unchanged in both. The enabled cache stores both text and condition-image prefix K/V at all 32 transformer layers. The first pass fills the cache; subsequent passes process only target tokens through the transformer blocks. No TaylorSeer, text-cache approximation, offload, quantization, or compilation is enabled.",
        "",
        f"The audited prefix has {cached_audit['prefix_tokens']:,} tokens and {cached_audit['cache_payload_bytes'] / 2**30:.4f} GiB of K/V tensors. The input/target latent layouts are `{cached_audit['img_shapes']}`; each spatial latent dimension corresponds to 16 pixels. The original reference is 1024 × 704; Qwen resizes conditioning images using `output_resolution=1024`, so its processed reference resolution differs from the earlier Flux experiment.",
        "",
        "Timing includes image preprocessing, prompt/image encoding, cache construction, denoising, VAE decoding, and PIL conversion. It excludes model loading, validation, PNG saving, and garbage collection. Memory includes all resident model weights and caches. The default segmented SDPA processor and automatic PyTorch backend selection are used in both modes, with no flex-attention compilation.",
        "",
        "## Outputs and validation",
        "",
        f"![Image comparison]({artifact_base}/comparison_seed_42.png)",
        "",
        "Both settings use identical initial noise and have matching first-step latents. In reduced precision, the different sequence layouts can change floating-point rounding and subsequent samples; exact output equality between settings is not assumed. Every warmup and timed output is saved and verified to match its own mode's validation image. The grid composites the native RGBA images on white; original PNGs preserve alpha.",
        "",
        "All 18 output PNG hashes and the preflight cache trace are verified by the report script. Final latents are saved for the two validation images. Tiny fp32 checks cover zero, one, and two reference images and two seeds, and verify that cached tensors own only prefix storage.",
        "",
        f"[Pixel differences]({artifact_base}/results/quality_metrics.json) · [Validation and cache audit]({artifact_base}/results/validation.json) · [Raw timings]({artifact_base}/results/runs.csv) · [Environment and exact prompt]({artifact_base}/results/metadata.json)",
        "",
        f"[HF Job]({job['job_url']}) · [Run instructions](../../README.md) · [Benchmark script](results/benchmark.py) · [SVG]({artifact_base}/qwenimage21_performance.svg) · [PDF]({artifact_base}/qwenimage21_performance.pdf)",
        "",
    ]
)
(run / "README.md").write_text(report)
shutil.copy2(base / "source_manifest.json", run / "source_manifest.json")
print("\n".join(table))
print(f"Verified all 18 generated images. Report: {run / 'README.md'}")
