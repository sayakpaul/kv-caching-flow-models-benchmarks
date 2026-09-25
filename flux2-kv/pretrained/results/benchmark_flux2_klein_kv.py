import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

import diffusers
from diffusers import Flux2KleinKVPipeline


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-9b-kv")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument(
        "--prompt", default="Dress this cat as a wizard wearing a blue hat and cloak, keeping its face and pose."
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = Image.open(args.reference).convert("RGB")
    reference.save(args.output / "reference.png")
    pipe = Flux2KleinKVPipeline.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.transformer.set_attention_backend("_native_flash")
    pipe.set_progress_bar_config(disable=True)
    metadata = {
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "prompt": args.prompt,
        "output_size": [args.size, args.size],
        "reference_size": list(reference.size),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "steps": args.steps,
        "warmups_per_mode_per_step_count": args.warmups,
        "measured_runs_per_mode_per_step_count": args.repeats,
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "platform": platform.platform(),
        "attention_backend": "Flux transformer: PyTorch SDPA Flash (_native_flash); text encoder and VAE: automatic SDPA selection",
        "pipeline_residency": "all components on GPU; no offload, compilation, or quantization",
        "latency_scope": "full pipeline call: prompt encoding, reference VAE encoding, denoising, VAE decoding, PIL conversion; excludes model loading and PNG serialization",
        "memory_scope": "peak torch.cuda allocated bytes during full pipeline call, including resident weights; allocator peaks reset before each call",
        "reserved_memory_note": "recorded for diagnostics only; both modes share a warmed allocator pool",
        "uncached_behavior": "fixed reference timestep and reference-only attention; reference states recomputed every step; no cache allocation or storage",
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    with torch.no_grad():
        for steps in args.steps:
            for phase, rounds in (("warmup", args.warmups), ("measure", args.repeats)):
                for run in range(rounds):
                    modes = (True, False) if run % 2 == 0 else (False, True)
                    for use_cache in modes:
                        mode = "cached" if use_cache else "uncached"
                        gc.collect()
                        torch.cuda.synchronize()
                        baseline = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        generator = torch.Generator(device="cuda").manual_seed(args.seed)
                        started = time.perf_counter()
                        output = pipe(
                            prompt=args.prompt,
                            image=reference,
                            height=args.size,
                            width=args.size,
                            num_inference_steps=steps,
                            generator=generator,
                            use_kv_cache=use_cache,
                            output_type="pil",
                        ).images[0]
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - started
                        peak = torch.cuda.max_memory_allocated()
                        record = {
                            "mode": mode,
                            "steps": steps,
                            "phase": phase,
                            "run": run + 1,
                            "seed": args.seed,
                            "seconds": seconds,
                            "baseline_allocated_bytes": baseline,
                            "peak_allocated_bytes": peak,
                            "incremental_peak_bytes": peak - baseline,
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "image": f"{steps}_steps/{mode}/{phase}_{run + 1:02d}_seed_{args.seed}.png",
                            "pixel_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                        }
                        image_path = args.output / record["image"]
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        output.save(image_path)
                        del output
                        records.append(record)
                        with (args.output / "runs.csv").open("w") as handle:
                            writer = csv.DictWriter(handle, fieldnames=list(record))
                            writer.writeheader()
                            writer.writerows(records)
                        print(json.dumps(record), flush=True)
    summary = []
    comparisons = []
    for steps in args.steps:
        for mode in ("uncached", "cached"):
            selected = [r for r in records if r["phase"] == "measure" and r["steps"] == steps and r["mode"] == mode]
            times = [r["seconds"] for r in selected]
            summary.append(
                {
                    "steps": steps,
                    "mode": mode,
                    "median_seconds": statistics.median(times),
                    "min_seconds": min(times),
                    "max_seconds": max(times),
                    "mean_seconds": statistics.mean(times),
                    "stdev_seconds": statistics.stdev(times),
                    "max_peak_allocated_gib": max(r["peak_allocated_bytes"] for r in selected) / 2**30,
                    "median_peak_allocated_gib": statistics.median(r["peak_allocated_bytes"] for r in selected)
                    / 2**30,
                    "max_incremental_peak_gib": max(r["incremental_peak_bytes"] for r in selected) / 2**30,
                    "unique_pixel_hashes": len({r["pixel_sha256"] for r in selected}),
                }
            )
        pair = [
            np.asarray(Image.open(args.output / f"{steps}_steps/{mode}/measure_01_seed_{args.seed}.png"))
            for mode in ("uncached", "cached")
        ]
        delta = pair[0].astype(np.float64) - pair[1].astype(np.float64)
        comparisons.append(
            {
                "steps": steps,
                "pixel_max_abs_difference_0_255": float(np.abs(delta).max()),
                "pixel_mean_abs_difference_0_255": float(np.abs(delta).mean()),
                "pixel_rmse_0_255": float(np.sqrt(np.mean(delta**2))),
                "identical_pixels_fraction": float(np.all(delta == 0, axis=-1).mean()),
            }
        )
    (args.output / "summary.json").write_text(
        json.dumps({"measurements": summary, "image_comparisons": comparisons}, indent=2) + "\n"
    )
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.subplots_adjust(top=0.75, bottom=0.20, wspace=0.3)
    fig.suptitle("FLUX.2 Klein 9B: KV caching", x=0.125, ha="left", fontsize=20, fontweight="bold")
    fig.text(
        0.125,
        0.83,
        f"{metadata['gpu']} · bfloat16 · {args.size} × {args.size} · one reference image\n{args.warmups} warmups + {args.repeats} measured runs per setting · seed {args.seed}",
        color="#475569",
    )
    x = np.arange(len(args.steps))
    for mode, shift, color, label in (
        ("uncached", -0.18, "#94a3b8", "Without cache"),
        ("cached", 0.18, "#2563eb", "With cache"),
    ):
        selected = [next(r for r in summary if r["steps"] == steps and r["mode"] == mode) for steps in args.steps]
        for ax, metric in zip(axes, ("median_seconds", "max_peak_allocated_gib")):
            values = [r[metric] for r in selected]
            bars = ax.bar(x + shift, values, width=0.34, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", padding=5)
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
        axes,
        ("Full-pipeline latency", "Peak GPU memory"),
        ("Seconds (median; whiskers = min–max)", "Allocated GPU memory (GiB)"),
    ):
        ax.set_title(title, loc="left", pad=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [f"{steps} steps" for steps in args.steps])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
        ax.legend(frameon=False, fontsize=9)
    fig.text(
        0.125,
        0.055,
        "Includes text/image encoding and image decoding. PNG saving is excluded. Memory includes resident weights.",
        fontsize=9,
        color="#475569",
    )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output / f"benchmark.{extension}", dpi=200, facecolor="white")
    print(json.dumps({"summary": summary, "image_comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
