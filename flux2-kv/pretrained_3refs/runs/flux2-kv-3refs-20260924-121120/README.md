# FLUX.2 Klein 9B KV benchmark with three reference images

Artifacts are hosted in the public [HF bucket](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks).

With three reference images, KV caching reduced median full-pipeline latency by **42.1% at 4 steps** and **51.3% at 8 steps**. Speedups were **1.73×** and **2.05×**, respectively. Peak allocated GPU memory increased by **4.124 GiB**. Cached and uncached output PNGs were pixel-identical at each step count.

| Measurement · three references | Without cache | With cache |
| --- | ---: | ---: |
| Median latency · 4 steps | 5.950 s | 3.446 s |
| Median latency · 8 steps | 11.459 s | 5.580 s |
| Peak GPU memory · both step counts | 35.45 GiB | 39.58 GiB |

The observed timing ranges across five measured runs were 5.9433–5.9556 s without caching versus 3.4412–3.4517 s with caching at 4 steps, and 11.4478–11.4726 s versus 5.5760–5.5869 s at 8 steps.

## Comparison with one reference image

| Measurement | One reference | Three references |
| --- | ---: | ---: |
| Cache speedup · 4 steps | 1.30× | 1.73× |
| Cache speedup · 8 steps | 1.40× | 2.05× |
| Latency reduction · 4 steps | 23.1% | 42.1% |
| Latency reduction · 8 steps | 28.4% | 51.3% |
| Extra peak memory from caching | 0.749 GiB | 4.124 GiB |

The larger reference sequence makes skipping repeated reference computation more beneficial. Its stored K/V are also larger: 4.125 GiB for three references, compared with 1.375 GiB for one. These sizes are `2 projections × 32 layers × reference_tokens × 4096 channels × 2 bytes`; the reference-token count increases from 2,816 to 8,448. Overall peak memory reflects all tensors live at the peak, rather than just the cache payload.

## Controlled setup

- [Completed three-reference HF Job](https://huggingface.co/jobs/sayakpaul/6ab513746b030d633f68e159).
- [Previous single-reference HF Job](https://huggingface.co/jobs/sayakpaul/6ab50f3a6b030d633f68e080).
- Checkpoint: `black-forest-labs/FLUX.2-klein-9b-kv`, revision `a6dfb36eca3a3906eb2fd460795adfb844e5fcce`.
- NVIDIA A100-SXM4-80GB; driver 580.178.04; PyTorch 2.10.0+cu128; CUDA 12.8; bfloat16.
- 1024 × 1024 output, batch size one; all pipeline components resident on GPU; no compilation, quantization, or offload.
- Flux uses PyTorch SDPA Flash (`_native_flash`); text encoder and VAE use automatic SDPA backend selection.
- Same prompt: `Dress this cat as a wizard wearing a blue hat and cloak, keeping its face and pose.`
- Generator reset to **seed 42 before every call**.
- **Three warmups, then five measured runs for each cache-mode/step-count combination.** Mode order alternates between cached-first and uncached-first.
- The installed package versions, GPU model, driver, model revision, prompt, seed, output resolution, and transformer/pipeline source patch were checked against the previous run. Reference count and the additional images are the workload changes. These are separate jobs on the same GPU model, not measurements on the same physical GPU allocation.

All three references are 1024 × 704. Reference 1 is the original `diffusers/cat.png`, with the same file hash as the single-reference benchmark. Reference 2 comes from `diffusers/sdxl_reference_input_cat.jpg` (another view/crop of the tuxedo cat), and reference 3 from `diffusers/wan-cat.jpg` (a cat wearing sunglasses). The latter two use aspect-preserving resize and center crop with `PIL.ImageOps.fit` and LANCZOS. Exact source URLs, preprocessing, dimensions, hashes, and input PNGs are included in the artifacts.

## Measurement and correctness

Latency covers the complete pipeline call, including text encoding, encoding all reference images, denoising, decoding, and conversion to a PIL image. CUDA is synchronized around the call. Downloading/loading the model and saving PNGs are excluded.

Peak memory is `torch.cuda.max_memory_allocated()`, reset before every call, including model weights. It is not total device usage from `nvidia-smi`. Both modes share the same warmed allocator; reserved memory is recorded only for diagnostics and is not used for the comparison. Every measured run had the same peak allocated memory within its mode.

The uncached baseline uses `use_kv_cache=False`: fixed reference timestep and reference-only attention are retained, but reference states are recomputed each step without allocating or storing a KV cache. It does not use `kv_cache_mode=None`, which changes the model's attention and modulation behavior. Cached execution uses the normal first-step extraction and later reuse; the cache is cleared at the end of each pipeline call.

All 32 saved output PNGs (including warmups) were downloaded, decoded, and checked against their logged pixel hashes. At each step count all 16 images across both modes and all repetitions share a single pixel hash. Cross-mode maximum pixel difference, mean absolute difference, and pixel RMSE are zero. This establishes equality of rendered images in this setup, not an across-backend guarantee or a measurement of latent differences.

Before launching, the tiny CPU float32 validation passed with zero, one, two, and three reference images. It checks exact cached/uncached latent equality and raises if the uncached path allocates a cache. Focused Ruff and formatting checks passed.

The performance comparison covers this set of images, prompt, seed, output size, and hardware/software configuration. The observed repetition ranges do not quantify variation across different prompts or reference sets.

## Files and reproduction

Results live in [runs/flux2-kv-3refs-20260924-121120/](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/tree/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120):

- [results/](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/tree/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/results): all output and reference PNGs; benchmark chart in PNG/SVG/PDF; [runs.csv](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/results/runs.csv); [summary.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/results/summary.json); environment metadata; source [patch](results/diffusers.patch) and [benchmark script](results/benchmark_flux2_klein_kv.py).
- [comparison/](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/tree/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/comparison): one-versus-three reference chart in PNG/SVG/PDF; output-image comparison; reference gallery; combined CSV and calculated speedups.
- [job.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/job.json): job URL, pinned dependencies, and private bucket volume paths.

The full source bundle is [inputs/diffusers-source.tar.gz](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/inputs/diffusers-source.tar.gz), based on Diffusers revision `e0118ade2f60234c41bacf40330a7e2f61108849`. Source hashes were verified against the executed job's manifest.

Apply the patch using the [three-reference instructions](../../README.md). The benchmark accepts multiple paths after `--reference`. The launcher supplies the three saved reference images. Run this from the repository root:

```bash
python flux2-kv/pretrained_3refs/submit_job.py
```

This submits a paid A100 job with the packaged source. Each submission creates fresh local and private bucket output paths; previous runs are preserved. It pins package versions using the existing single-reference run's [metadata.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/metadata.json). The latest submitted job is recorded in [job.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/job.json); each run also retains its own record. `retrieve_results.py` downloads the latest submitted run, and `compare_results.py` verifies the inputs/outputs and generates the comparison with the single-reference results.
