# FLUX.2 Klein 9B KV: cache benchmark

Artifacts are hosted in the public [HF bucket](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks). Before running the commands below, restore them by running `python download_artifacts.py` from the repository root; see the [download instructions](../../../README.md).

KV caching reduced median full-pipeline latency by **23.1% at 4 steps** and **28.4% at 8 steps** on one NVIDIA A100-SXM4-80GB. Peak allocated GPU memory increased by **0.749 GiB**. Cached and uncached output images were pixel-identical for this input and seed.

| Denoising steps | Cache | Median latency | Observed min–max | Peak allocated GPU memory |
| --- | --- | ---: | ---: | ---: |
| 4 | Off | 3.0852 s | 3.0823–3.0913 s | 34.7348 GiB |
| 4 | On | 2.3738 s | 2.3706–2.3747 s | 35.4835 GiB |
| 8 | Off | 5.8830 s | 5.8766–5.8875 s | 34.7348 GiB |
| 8 | On | 4.2094 s | 4.2040–4.2125 s | 35.4835 GiB |

Speedup is uncached median latency divided by cached median latency: **1.2997×** and **1.3976×**, respectively. The range describes five repeated measurements; it is not a confidence interval across prompts or machines.

## Setup

- Actual checkpoint: [black-forest-labs/FLUX.2-klein-9b-kv](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv), pinned revision `a6dfb36eca3a3906eb2fd460795adfb844e5fcce`.
- [Completed HF Job](https://huggingface.co/jobs/sayakpaul/6ab50f3a6b030d633f68e080), hardware flavor `a100-large`.
- NVIDIA A100-SXM4-80GB, driver 580.178.04, PyTorch 2.10.0+cu128, CUDA 12.8, bfloat16.
- All pipeline components resident on GPU; no quantization, compilation, or CPU offload.
- Flux transformer uses PyTorch SDPA Flash (`_native_flash`). Text encoder and VAE use automatic SDPA backend selection.
- One unmodified 1024 × 704 [reference image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png), saved as [results/reference.png](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/reference.png). Output is 1024 × 1024, batch size one.
- Prompt: `Dress this cat as a wizard wearing a blue hat and cloak, keeping its face and pose.`
- Generator reset to seed **42 on every invocation**, for both modes and both step counts.
- **Three warmups followed by five measured runs for every mode/step-count combination.** Mode order alternates between cached-first and uncached-first. Warmup measurements are retained in the CSV but excluded from the reported results.

## What was measured

Timing covers the complete pipeline invocation: prompt encoding, reference-image VAE encoding, denoising, VAE decoding, and conversion to a PIL image. CUDA is synchronized before and after the call. Model downloading/loading and PNG serialization are outside the timed region.

Peak memory is `torch.cuda.max_memory_allocated()`, reset before each invocation. It includes all resident model weights and active PyTorch GPU tensors. It is not total device usage reported by `nvidia-smi`. Reserved allocator memory is recorded separately for diagnostics; both modes share the same warmed allocator pool, so reserved memory is not used for the comparison. All measured runs had the same peak allocated memory within each mode.

Reference K/V storage alone is 1.375 GiB for this input: `2 projections × 32 layers × 2816 reference tokens × 4096 hidden dimensions × 2 bytes`. The observed change in overall peak memory is smaller because a peak measures simultaneously live tensors across the whole pipeline, not the sum of independently measured allocations.

These results cover one reference image, one prompt, one seed, and this software/hardware configuration. They do not estimate multi-reference performance or variation across inputs.

## Uncached baseline

The experimental Diffusers patch adds `use_kv_cache=False` to `Flux2KleinKVPipeline.__call__` and a transformer `"recompute"` mode. It recomputes the reference tokens on every denoising step while retaining the fixed reference timestep and reference-only attention. It allocates no `Flux2KVCache` and does not clone or store reference K/V.

Using `kv_cache_mode=None` would change the reference attention and timestep behavior, so that is not the baseline used here. The cached mode remains the normal first-step extraction followed by per-layer K/V reuse. Its cache is cleared at the end of every pipeline invocation; no cache is reused between benchmark runs.

Before submission, a tiny CPU float32 pipeline produced exactly equal cached/uncached final latents with zero, one, and two reference images. The validation raises an error if the uncached path attempts to allocate a cache. Focused Ruff checks and formatting checks passed.

The first HF Job failed before completing a warmup because a global Flash-only SDPA setting was incompatible with the Qwen text encoder's mask. It produced no benchmark measurements. The completed job fixes Flash Attention only for the Flux transformer; its text encoder and VAE keep normal backend selection. [failed_job_1.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/failed_job_1.json) records that attempt.

## Images and data

- [results/benchmark.png](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/benchmark.png), [SVG](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/benchmark.svg), and [PDF](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/benchmark.pdf): performance chart.
- [results/image_comparison.png](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/image_comparison.png): visual comparison of the first measured output in each setting.
- [4-step outputs](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/tree/flux2-kv/pretrained/results/4_steps) and [8-step outputs](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/tree/flux2-kv/pretrained/results/8_steps): all 32 output PNGs, including warmups.
- [results/runs.csv](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/runs.csv): every invocation's time, memory, seed, output path, and decoded-pixel SHA-256.
- [results/summary.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/summary.json): aggregates and image differences.
- [results/metadata.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/metadata.json): full environment, dependency versions, inputs, and measurement scope.
- [results/source_manifest.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/source_manifest.json): source revision and hashes.

All 32 downloaded PNGs were decoded and their pixel hashes checked against the run log. For each step count, all 16 outputs share one pixel hash across both modes and all repetitions. Cross-mode pixel maximum difference, mean absolute difference, and RMSE are all zero. This establishes image equality in this experiment, not exact latent equality on the pretrained checkpoint or a guarantee across other backends.

## Code and reproduction

Apply the patch using the [one-reference instructions](../README.md). After restoring the artifacts, run this from the repository root on a suitable GPU with checkpoint access:

```bash
python flux2-kv/pretrained/inputs/benchmark_flux2_klein_kv.py \
  --revision a6dfb36eca3a3906eb2fd460795adfb844e5fcce \
  --reference flux2-kv/pretrained/inputs/reference.png \
  --output /tmp/flux2-kv-benchmark-reproduction
```

Default arguments reproduce the 4/8-step cases, seed 42, three warmups, five measured runs, prompt, and output size. Matching timings also requires the recorded GPU and software environment.

[inputs/diffusers-source.tar.gz](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/inputs/diffusers-source.tar.gz) contains the exact modified source and benchmark used remotely, based on Diffusers revision `e0118ade2f60234c41bacf40330a7e2f61108849`. The readable patch and standalone benchmark script are also included. `run_job.py`, `submit_job.py`, and [job.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/job.json) record how the job was launched and how its private bucket volumes were mounted. Choose fresh output paths before submitting another paid run.
