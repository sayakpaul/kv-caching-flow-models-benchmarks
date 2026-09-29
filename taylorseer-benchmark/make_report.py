import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
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
    f"taylorseer-benchmark/runs/{run.name}"
)
summary = json.loads((results / "summary.json").read_text())
validation = json.loads((results / "validation.json").read_text())
rows = list(csv.DictReader((results / "runs.csv").open()))
modes = ("reference_only", "reference_and_taylorseer")
lookup = {(x["steps"], x["mode"]): x for x in summary}
assert len(rows) == 32 and len(validation) == 12
for row in [*rows, *validation]:
    image = Image.open(results / row["image"])
    assert hashlib.sha256(image.tobytes()).hexdigest() == row["pixel_sha256"]
for item in validation:
    audit = item["audit"]
    assert audit["transformer"][0]["populated_reference_layers"] == 32
    assert len(audit["transformer"]) == item["steps"]
    assert all(x["mode"] == "cached" for x in audit["transformer"][1:])
    if item["mode"] == modes[1]:
        assert len(audit["modules"]) == item["steps"] * 32
        for name in {x["module"] for x in audit["modules"]}:
            assert [x["step"] for x in audit["modules"] if x["module"] == name and not x["compute"]] == list(
                range(4, item["steps"] + 1, 2)
            )
        baseline = next(
            x
            for x in validation
            if x["mode"] == modes[0] and x["seed"] == item["seed"] and x["steps"] == item["steps"]
        )
        assert baseline["first_latent_sha256"] == item["first_latent_sha256"]
for item in summary:
    group = [x for x in rows if int(x["steps"]) == item["steps"] and x["mode"] == item["mode"]]
    assert sum(x["phase"] == "warmup" for x in group) == 3
    assert sum(x["phase"] == "measure" for x in group) == 5
    expected = next(
        x["pixel_sha256"]
        for x in validation
        if x["steps"] == item["steps"] and x["mode"] == item["mode"] and x["seed"] == 42
    )
    assert all(x["pixel_sha256"] == expected for x in group)
changes = [
    {
        "steps": step,
        "latency_reduction_percent": 100
        * (1 - lookup[(step, modes[1])]["median_seconds"] / lookup[(step, modes[0])]["median_seconds"]),
    }
    for step in (4, 8)
]
table = ["| Measurement | Reference KV cache | + TaylorSeer |", "| --- | ---: | ---: |"]
for change in changes:
    step = change["steps"]
    a, b = [lookup[(step, mode)] for mode in modes]
    table.append(
        f"| Latency ({step} steps) | {a['median_seconds']:.3f}s | {b['median_seconds']:.3f}s ({change['latency_reduction_percent']:.1f}% lower) |"
    )
if all(lookup[(4, mode)]["max_peak_allocated_gib"] == lookup[(8, mode)]["max_peak_allocated_gib"] for mode in modes):
    table.append(
        f"| Peak GPU memory | {lookup[(4, modes[0])]['max_peak_allocated_gib']:.2f} GiB | {lookup[(4, modes[1])]['max_peak_allocated_gib']:.2f} GiB |"
    )
else:
    for step in (4, 8):
        table.append(
            f"| Peak GPU memory ({step} steps) | {lookup[(step, modes[0])]['max_peak_allocated_gib']:.2f} GiB | {lookup[(step, modes[1])]['max_peak_allocated_gib']:.2f} GiB |"
        )
(run / "table.md").write_text("\n".join(table) + "\n")
(run / "changes.json").write_text(json.dumps(changes, indent=2))


def font(size, bold=False):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf", size)


for seed in (42, 43, 44):
    canvas = Image.new("RGB", (1200, 1410), "#F7F9FC")
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 26), "Reference KV caching + TaylorSeer", font=font(34, True), fill="#18283C")
    draw.text((48, 80), f"FLUX.2 Klein 9B KV · seed {seed} · same prompt and reference", font=font(21), fill="#5C6D81")
    for column, label in enumerate(("Reference KV cache", "+ TaylorSeer")):
        draw.text((48 + column * 576, 139), label, font=font(26, True), fill="#18283C")
    for row, step in enumerate((4, 8)):
        y = 189 + row * 598
        draw.text((48, y), f"{step} denoising steps", font=font(22), fill="#5C6D81")
        for column, mode in enumerate(modes):
            sample = Image.open(results / f"quality/{step}_steps/seed_{seed}/{mode}.png").convert("RGB")
            canvas.paste(sample.resize((528, 528), Image.Resampling.LANCZOS), (48 + column * 576, y + 39))
    canvas.save(run / f"comparison_seed_{seed}.png")

plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.70, wspace=0.28)
fig.text(0.08, 0.94, "Combining reference KV caching and TaylorSeer", fontsize=20, weight="bold")
fig.text(
    0.08,
    0.88,
    "FLUX.2 Klein 9B KV · A100 80 GB · bfloat16 · 1024 × 1024 · one reference",
    fontsize=11,
    color="#526072",
)
for ax, field, title, unit in zip(
    axes, ("median_seconds", "max_peak_allocated_gib"), ("End-to-end latency", "Peak GPU memory"), ("Seconds", "GiB")
):
    x = np.arange(2)
    for index, (mode, label, color) in enumerate(
        zip(modes, ("Reference KV cache", "+ TaylorSeer"), ("#3366CC", "#008B80"))
    ):
        values = [lookup[(step, mode)][field] for step in (4, 8)]
        bars = ax.bar(x + (index - 0.5) * 0.34, values, 0.31, label=label, color=color, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=6)
        if field == "median_seconds":
            errors = [
                [lookup[(step, mode)][field] - lookup[(step, mode)]["min_seconds"] for step in (4, 8)],
                [lookup[(step, mode)]["max_seconds"] - lookup[(step, mode)][field] for step in (4, 8)],
            ]
            ax.errorbar(
                x + (index - 0.5) * 0.34,
                values,
                yerr=errors,
                fmt="none",
                ecolor="#263443",
                capsize=3,
                linewidth=1,
                zorder=4,
            )
    ax.set_xticks(x, ("4 steps", "8 steps"))
    ax.set_title(title, loc="left", fontsize=14, pad=14)
    ax.set_ylabel(unit)
    ax.set_ylim(0, max(item[field] for item in summary) * 1.2)
    ax.grid(axis="y", color="#E3E8EE", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#CDD5DE")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.071, 0.84), frameon=False, ncol=2)
fig.text(
    0.08,
    0.115,
    "3 warmups + 5 timed runs per setting · median latency; whiskers show min–max",
    fontsize=10,
    color="#526072",
)
fig.text(
    0.08,
    0.073,
    "Peak allocated memory includes weights. TaylorSeer: order 1, interval 2, first 3 steps computed.",
    fontsize=10,
    color="#526072",
)
for extension in ("png", "svg", "pdf"):
    fig.savefig(run / f"taylorseer_performance.{extension}", dpi=220, facecolor="white")
plt.close(fig)
report = "\n".join(
    [
        "# Reference KV caching + TaylorSeer",
        "",
        *table,
        "",
        "A100-SXM4-80GB, bfloat16, 1024 × 1024 output, one cat reference, seed 42. Each setting has three warmups followed by five timed calls, with alternating mode order. The baseline is measured again in this job. Text KV caching is disabled in both cases.",
        "",
        "Latency is the median of five complete pipeline calls: prompt/reference encoding, cache construction, denoising, decoding, PIL conversion, and cache reset. It excludes model loading, saving PNGs, garbage collection, and installing/removing TaylorSeer hooks. Peak memory is the largest peak allocated GPU memory across those five calls, including weights and caches. No offload, compilation, or quantization.",
        "",
        f"![Performance]({artifact_base}/taylorseer_performance.png)",
        "",
        "## Configuration",
        "",
        "TaylorSeer uses the built-in default attention-module hooks in all 8 double and 24 single blocks, with `cache_interval=2`, `disable_cache_before_step=3`, `max_order=1`, and bfloat16 factors. Single-stream attention modules also include the parallel MLP computation. The first three steps compute fully. At four steps, step 4 is predicted; at eight steps, steps 4, 6, and 8 are predicted. All other steps compute and update the factors.",
        "",
        'The three initial full steps allow the attention output shapes to stabilize after reference tokens leave the live sequence. The pinned implementation restarts Taylor factors on a shape change. Each pipeline call runs inside `transformer.cache_context("cond")`; the pipeline clears stateful hooks at the end. No Diffusers patch is needed at this revision.',
        "",
        "```python",
        "from diffusers import TaylorSeerCacheConfig",
        "",
        "pipe.transformer.enable_cache(TaylorSeerCacheConfig(",
        "    cache_interval=2,",
        "    disable_cache_before_step=3,",
        "    max_order=1,",
        "    taylor_factors_dtype=torch.bfloat16,",
        "))",
        'with pipe.transformer.cache_context("cond"):',
        "    image = pipe(image=reference, prompt=prompt, num_inference_steps=8,",
        '                 generator=torch.Generator("cuda").manual_seed(42)).images[0]',
        "```",
        "",
        "## Image comparisons and validation",
        "",
        "TaylorSeer is approximate and changes the generated images. These parameters were chosen before viewing results; this is one prompt/reference pair, not a general quality benchmark. All first-step latents match the baseline. Audits confirm all 32 reference caches are populated and all TaylorSeer hooks follow the stated schedule. Tiny-model checks cover shape changes, exact no-prediction controls, and repeatability across calls.",
        "",
        "Visual inspection: the wizard-cat composition remains recognizable in all six comparisons. At four steps, all three seeds show noticeably rougher, oversharpened-looking fur and fabric, with stronger background texture. At eight steps the differences are less pronounced, but texture and contrast still change. The measured speedup is therefore not quality-neutral.",
        "",
        f"![Seed 42]({artifact_base}/comparison_seed_42.png)",
        "",
        f"[Seed 43]({artifact_base}/comparison_seed_43.png) · [Seed 44]({artifact_base}/comparison_seed_44.png) · [Pixel differences]({artifact_base}/results/quality_metrics.json)",
        "",
        "All 32 warmup/timed PNGs plus 12 quality PNGs are saved and their hashes verified. Final quality latents are saved alongside each image. Every timed/warmup output matches its seed-42 quality image exactly.",
        "",
        f"[HF Job]({job['job_url']}) · [Launch instructions](../../README.md) · [Measurement script](results/benchmark.py) · [Configuration and audit](results/config.py)",
        "",
        f"[Raw runs]({artifact_base}/results/runs.csv) · [Summary]({artifact_base}/results/summary.json) · [Environment and prompt]({artifact_base}/results/metadata.json) · [Validation]({artifact_base}/results/validation.json) · [SVG]({artifact_base}/taylorseer_performance.svg) · [PDF]({artifact_base}/taylorseer_performance.pdf)",
        "",
    ]
)
(run / "README.md").write_text(report)
shutil.copy2(base / "source_manifest.json", run / "source_manifest.json")
print("\n".join(table))
print(f"Verified all 44 generated images. Report: {run / 'README.md'}")
