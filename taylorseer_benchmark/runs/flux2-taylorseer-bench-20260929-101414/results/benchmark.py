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
from config import audit_cache, cache_config, check_audit
from PIL import Image

from diffusers import Flux2KleinKVPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = "black-forest-labs/FLUX.2-klein-9b-kv"
    revision = "a6dfb36eca3a3906eb2fd460795adfb844e5fcce"
    prompt = "Dress this cat as a wizard wearing a blue hat and cloak, keeping its face and pose."
    reference = Image.open(args.reference).convert("RGB")
    reference.save(out / "reference.png")
    pipe = Flux2KleinKVPipeline.from_pretrained(model, revision=revision, torch_dtype=torch.bfloat16).to("cuda")
    pipe.transformer.set_attention_backend("_native_flash")
    pipe.set_progress_bar_config(disable=True)
    kwargs = {"prompt": prompt, "image": reference, "height": 1024, "width": 1024}
    modes = ("reference_only", "reference_and_taylorseer")
    metadata = {
        "model": model,
        "revision": revision,
        "source": json.loads((Path(__file__).parent / "source_info.json").read_text()),
        "prompt": prompt,
        "seed": 42,
        "quality_seeds": [42, 43, 44],
        "steps": [4, 8],
        "warmups": 3,
        "measured_repeats": 5,
        "output_size": [1024, 1024],
        "reference_size": list(reference.size),
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "modes": list(modes),
        "taylorseer": {
            "cache_interval": 2,
            "disable_cache_before_step": 3,
            "max_order": 1,
            "taylor_factors_dtype": "bfloat16",
            "modules": "Default attention modules in all double/single blocks",
            "prediction_steps_1_based": {"4": [4], "8": [4, 6, 8]},
        },
        "timing_scope": "Full pipeline: text/reference encoding, cache construction, denoising, decoding, PIL conversion, and state reset. Excludes model load, validation, PNG saving, GC, hook installation and removal, and context entry/exit.",
        "memory_scope": "Peak torch.cuda allocated memory per pipeline call including resident model weights and caches. Reserved memory is diagnostic only.",
        "reference_cache": "Enabled in both modes. No text KV caching.",
        "backend": "Flux: _native_flash; text encoder and VAE: automatic SDPA",
        "offload_compile_quantization": False,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    expected, first_steps, validation = {}, {}, []
    for steps in (4, 8):
        for seed in (42, 43, 44):
            for mode in modes:
                captured, audit = {}, {}

                def capture(pipeline, step, timestep, values):
                    if step in (0, steps - 1):
                        captured[step] = values["latents"].detach().float().cpu()
                    return values

                if mode == modes[1]:
                    pipe.transformer.enable_cache(cache_config())
                with pipe.transformer.cache_context("cond"), audit_cache(pipe.transformer, audit):
                    image = pipe(
                        **kwargs,
                        num_inference_steps=steps,
                        generator=torch.Generator("cuda").manual_seed(seed),
                        callback_on_step_end=capture,
                    ).images[0]
                if mode == modes[1]:
                    check_audit(audit, steps, 32)
                    pipe.transformer.disable_cache()
                assert all(torch.isfinite(x).all() for x in captured.values())
                first_hash = hashlib.sha256(captured[0].contiguous().numpy().tobytes()).hexdigest()
                if mode == modes[0]:
                    first_steps[(steps, seed)] = first_hash
                assert first_hash == first_steps[(steps, seed)]
                pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                expected[(steps, mode, seed)] = pixel_hash
                path = f"quality/{steps}_steps/seed_{seed}/{mode}.png"
                (out / path).parent.mkdir(parents=True, exist_ok=True)
                image.save(out / path)
                torch.save(captured[steps - 1], (out / path).with_suffix(".pt"))
                record = {
                    "steps": steps,
                    "seed": seed,
                    "mode": mode,
                    "image": path,
                    "pixel_sha256": pixel_hash,
                    "first_latent_sha256": first_hash,
                    "final_latent_sha256": hashlib.sha256(
                        captured[steps - 1].contiguous().numpy().tobytes()
                    ).hexdigest(),
                    "audit": audit,
                }
                validation.append(record)
                (out / "validation.json").write_text(json.dumps(validation, indent=2))
                print("VALIDATED", steps, seed, mode, pixel_hash, flush=True)
                captured.clear()
                del image
    records = []
    for steps in (4, 8):
        for phase, count in (("warmup", 3), ("measure", 5)):
            for run in range(1, count + 1):
                ordered = modes if run % 2 else modes[::-1]
                for mode in ordered:
                    if mode == modes[1]:
                        pipe.transformer.enable_cache(cache_config())
                    gc.collect()
                    generator = torch.Generator("cuda").manual_seed(42)
                    with pipe.transformer.cache_context("cond"):
                        torch.cuda.synchronize()
                        baseline = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                        image = pipe(**kwargs, num_inference_steps=steps, generator=generator).images[0]
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                        peak = torch.cuda.max_memory_allocated()
                        reserved = torch.cuda.max_memory_reserved()
                    if mode == modes[1]:
                        pipe.transformer.disable_cache()
                    path = f"{steps}_steps/{mode}/{phase}_{run:02d}_seed_42.png"
                    (out / path).parent.mkdir(parents=True, exist_ok=True)
                    image.save(out / path)
                    pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                    assert pixel_hash == expected[(steps, mode, 42)]
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
    for steps in (4, 8):
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
    for steps in (4, 8):
        for seed in (42, 43, 44):
            directory = out / f"quality/{steps}_steps/seed_{seed}"
            a, b = [np.asarray(Image.open(directory / f"{mode}.png"), dtype=np.float32) for mode in modes]
            quality.append({"steps": steps, "seed": seed, "pixel_mae_0_255": float(np.abs(a - b).mean())})
    (out / "quality_metrics.json").write_text(json.dumps(quality, indent=2))
    print("TAYLORSEER_BENCHMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
