# Qwen-Image-2.1: KV caching benchmark

[Results](runs/qwenimage21-kv-bench-20260929-111910/README.md) · [Image comparison](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/runs/qwenimage21-kv-bench-20260929-111910/comparison_seed_42.png) · [Performance chart](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/runs/qwenimage21-kv-bench-20260929-111910/qwenimage21_performance.png)

Images, saved latents, result data, logs, and the source archive are in the public [HF bucket](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/tree/qwenimage21-benchmark).

This benchmarks `QwenImage21Pipeline` with `use_kv_cache=False` and `use_kv_cache=True`. Both retain `causal_condition=True`, so conditioning timesteps and attention rules are unchanged. The built-in cache stores both text and condition-image prefix K/V.

The requested setup is **40 steps**, A100 80 GB, bfloat16, 1024 × 1024 output, one cat reference, and seed 42. Each mode has three warmups followed by five measured calls. Mode order alternates. Latency covers the complete pipeline, and peak allocated memory includes model weights. All outputs are saved.

## Run with HF Jobs

```bash
# From the repository root.
cd qwenimage21-benchmark
python submit_job.py
```

The launcher creates a paid A100 job with pinned dependencies and unique artifact paths. It uses your authenticated Hugging Face token for model and Jobs access.

After completion:

```bash
python retrieve_results.py
python make_report.py
```

[job.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/job.json) identifies the latest run. Its report contains the compact latency/memory table, PNG/SVG/PDF chart, and output comparison without a reference thumbnail. Original PNGs preserve RGBA; the comparison grid composites them on white.

## Source and settings

No source patch is required. The job uses an unmodified archive of Diffusers commit `5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2`. Do not apply the earlier [Flux-specific patch](../flux2-kv/pretrained/inputs/diffusers.patch) here.

The model is `Qwen/Qwen-Image-2.1`, revision `790c92633540aa0cb11d9abf19eb46d861714758`. The job pins PyTorch 2.10.0, Transformers 5.17.0, tokenizers 0.23.1, and torchvision 0.25.0; complete dependencies are in [job.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/job.json) and `requirements.lock`.

- [inputs/settings.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/inputs/settings.json): exact prompt, seed, step count, resolution, and model revision.
- `inputs/benchmark.py`: validation, warmups, timing, peak memory, and serialization.
- `inputs/audit.py`: verifies cache modes, prefix length, payload, and tensor ownership outside timed runs.
- `inputs/validate_cache.py`: tiny fp32 model checks before loading the checkpoint.
- `run_job.py`: remote entry point.
- [source_manifest.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/source_manifest.json): input hashes.

No CFG, offload, quantization, compilation, TaylorSeer, or additional approximate caching is enabled. Attention uses the default segmented SDPA processor and automatic PyTorch backend selection in both cases.

The same original 1024 × 704 cat reference is used as in the Flux experiments. Qwen resizes condition images to its `output_resolution=1024` area budget, while output height and width are explicitly 1024. The actual latent layouts and cache sizes are recorded in [validation.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/runs/qwenimage21-kv-bench-20260929-111910/results/validation.json).

Cached and uncached outputs can differ in bf16 because the changed sequence layout changes floating-point rounding. The benchmark verifies identical first-step latents and repeatability within each mode, and saves the output pair for inspection.

To run locally, install the pinned Diffusers revision in a separate checkout with the recorded dependencies, then run:

```bash
python inputs/validate_cache.py
python inputs/benchmark.py --reference inputs/reference.png --output local-results
```
