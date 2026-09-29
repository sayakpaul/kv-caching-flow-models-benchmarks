import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from audit import audit_cache
from PIL import Image

from diffusers import QwenImage21Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads((Path(__file__).parent / "settings.json").read_text())
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = Image.open(args.reference).convert("RGB")
    reference.save(out / "reference.png")
    pipe = QwenImage21Pipeline.from_pretrained(
        settings["model"], revision=settings["revision"], torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    assert pipe.transformer.config.causal_condition
    kwargs = {k: settings[k] for k in ("prompt", "height", "width", "output_resolution", "true_cfg_scale")}
    kwargs["image"] = reference
    modes = ("without_cache", "with_cache")
    metadata = {
        **settings,
        "source": json.loads((Path(__file__).parent / "source_info.json").read_text()),
        "reference_size": list(reference.size),
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_cuda": torch.version.cuda,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "modes": list(modes),
        "attention_processor": type(pipe.transformer.transformer_blocks[0].attn.processor).__name__,
        "backend": "Default segmented SDPA processor, automatic PyTorch SDPA backend selection; no flex attention or compilation",
        "timing_scope": "Complete pipeline including image preprocessing, prompt/image encoding, cache construction, denoising, VAE decoding and PIL conversion. Excludes model load, validation, PNG saving and GC.",
        "memory_scope": "Peak torch.cuda allocated memory per complete pipeline call, including resident weights and caches. Reserved memory is diagnostic only.",
        "cache_scope": "Built-in text and condition-image prefix KV cache; use_kv_cache toggled, causal_condition remains True in both cases.",
        "offload_compile_quantization": False,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    expected, first_steps, validation = {}, {}, []
    for steps in settings["steps"]:
        for mode in modes:
            captured, audit = {}, []

            def capture(pipeline, step, timestep, values):
                if step in (0, steps - 1):
                    captured[step] = values["latents"].detach().float().cpu()
                return values

            with audit_cache(pipe.transformer, audit):
                image = pipe(
                    **kwargs,
                    num_inference_steps=steps,
                    use_kv_cache=mode == "with_cache",
                    generator=torch.Generator("cuda").manual_seed(settings["seed"]),
                    callback_on_step_end=capture,
                ).images[0]
            assert len(audit) == steps
            assert all(x["causal_condition"] for x in audit)
            if mode == "with_cache":
                assert audit[0]["mode"] == "extract"
                assert all(x["mode"] == "cached" and x["output_tokens"] == 4096 for x in audit[1:])
                assert all(x["cache_layers"] == 32 for x in audit)
            else:
                assert all(x["mode"] is None and x["cache_layers"] == 0 for x in audit)
            assert all(torch.isfinite(x).all() for x in captured.values())
            first_hash = hashlib.sha256(captured[0].contiguous().numpy().tobytes()).hexdigest()
            if mode == "without_cache":
                first_steps[steps] = first_hash
            assert first_hash == first_steps[steps]
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            expected[(steps, mode)] = pixel_hash
            path = f"quality/{steps}_steps/{mode}_seed_{settings['seed']}.png"
            (out / path).parent.mkdir(parents=True, exist_ok=True)
            image.save(out / path)
            torch.save(captured[steps - 1], (out / path).with_suffix(".pt"))
            record = {
                "steps": steps,
                "mode": mode,
                "image": path,
                "image_mode": image.mode,
                "pixel_sha256": pixel_hash,
                "first_latent_sha256": first_hash,
                "final_latent_sha256": hashlib.sha256(captured[steps - 1].contiguous().numpy().tobytes()).hexdigest(),
                "audit": audit,
            }
            validation.append(record)
            (out / "validation.json").write_text(json.dumps(validation, indent=2))
            print("VALIDATED", steps, mode, pixel_hash, audit[0], flush=True)
            captured.clear()
            del image
    records = []
    for steps in settings["steps"]:
        for phase, count in (("warmup", settings["warmups"]), ("measure", settings["measured_repeats"])):
            for run in range(1, count + 1):
                ordered = modes if run % 2 else modes[::-1]
                for mode in ordered:
                    gc.collect()
                    generator = torch.Generator("cuda").manual_seed(settings["seed"])
                    torch.cuda.synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    image = pipe(
                        **kwargs, num_inference_steps=steps, use_kv_cache=mode == "with_cache", generator=generator
                    ).images[0]
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    peak = torch.cuda.max_memory_allocated()
                    reserved = torch.cuda.max_memory_reserved()
                    path = f"{steps}_steps/{mode}/{phase}_{run:02d}_seed_{settings['seed']}.png"
                    (out / path).parent.mkdir(parents=True, exist_ok=True)
                    image.save(out / path)
                    pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                    assert pixel_hash == expected[(steps, mode)]
                    record = {
                        "steps": steps,
                        "mode": mode,
                        "phase": phase,
                        "run": run,
                        "seconds": elapsed,
                        "baseline_allocated_bytes": baseline,
                        "peak_allocated_gib": peak / 2**30,
                        "peak_reserved_gib": reserved / 2**30,
                        "image": path,
                        "pixel_sha256": pixel_hash,
                    }
                    records.append(record)
                    with (out / "runs.csv").open("w") as handle:
                        writer = csv.DictWriter(handle, fieldnames=record)
                        writer.writeheader()
                        writer.writerows(records)
                    print(json.dumps(record), flush=True)
                    del image
    summary = []
    for steps in settings["steps"]:
        for mode in modes:
            rows = [x for x in records if x["steps"] == steps and x["mode"] == mode and x["phase"] == "measure"]
            values = [x["seconds"] for x in rows]
            summary.append(
                {
                    "steps": steps,
                    "mode": mode,
                    "median_seconds": statistics.median(values),
                    "min_seconds": min(values),
                    "max_seconds": max(values),
                    "max_peak_allocated_gib": max(x["peak_allocated_gib"] for x in rows),
                }
            )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    quality = []
    for steps in settings["steps"]:
        directory = out / f"quality/{steps}_steps"
        a, b = [
            np.asarray(Image.open(directory / f"{mode}_seed_{settings['seed']}.png"), dtype=np.float32)
            for mode in modes
        ]
        quality.append(
            {
                "steps": steps,
                "pixel_mae_0_255_all_channels": float(np.abs(a - b).mean()),
                "pixels_identical": bool(np.array_equal(a, b)),
            }
        )
    (out / "quality_metrics.json").write_text(json.dumps(quality, indent=2))
    print("QWENIMAGE21_BENCHMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
