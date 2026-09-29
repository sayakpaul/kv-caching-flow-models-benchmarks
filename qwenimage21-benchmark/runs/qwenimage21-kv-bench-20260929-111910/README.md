# Qwen-Image-2.1: with and without KV caching

| Measurement | Without KV cache | With KV cache |
| --- | ---: | ---: |
| Latency (40 steps) | 32.202s | 17.798s (44.7% lower) |
| Peak GPU memory | 36.81 GiB | 38.89 GiB |

A100-SXM4-80GB, bfloat16, batch size 1, 1024 × 1024 output, 40 denoising steps, one cat reference, seed 42, no classifier-free guidance. Each mode has three warmups followed by five timed calls, with alternating mode order. The table uses median end-to-end latency and the largest peak allocated GPU memory across the five measured calls.

![Performance](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/runs/qwenimage21-kv-bench-20260929-111910/qwenimage21_performance.png)

## What is compared

The pipeline's built-in `use_kv_cache=False` and `use_kv_cache=True` settings. `causal_condition=True` remains unchanged in both. The enabled cache stores both text and condition-image prefix K/V at all 32 transformer layers. The first pass fills the cache; subsequent passes process only target tokens through the transformer blocks. No TaylorSeer, text-cache approximation, offload, quantization, or compilation is enabled.

The audited prefix has 4,246 tokens and 2.0732 GiB of K/V tensors. The input/target latent layouts are `[[[1, 54, 78], [1, 64, 64]]]`; each spatial latent dimension corresponds to 16 pixels. The original reference is 1024 × 704; Qwen resizes conditioning images using `output_resolution=1024`, so its processed reference resolution differs from the earlier Flux experiment.

Timing includes image preprocessing, prompt/image encoding, cache construction, denoising, VAE decoding, and PIL conversion. It excludes model loading, validation, PNG saving, and garbage collection. Memory includes all resident model weights and caches. The default segmented SDPA processor and automatic PyTorch backend selection are used in both modes, with no flex-attention compilation.

## Outputs and validation

![Image comparison](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/runs/qwenimage21-kv-bench-20260929-111910/comparison_seed_42.png)

Both settings use identical initial noise and have matching first-step latents. In reduced precision, the different sequence layouts can change floating-point rounding and subsequent samples; exact output equality between settings is not assumed. Every warmup and timed output is saved and verified to match its own mode's validation image. The grid composites the native RGBA images on white; original PNGs preserve alpha.

All 18 output PNG hashes and the preflight cache trace are verified by the report script. Final latents are saved for the two validation images. Tiny fp32 checks cover zero, one, and two reference images and two seeds, and verify that cached tensors own only prefix storage.

[Pixel differences](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/runs/qwenimage21-kv-bench-20260929-111910/results/quality_metrics.json) · [Validation and cache audit](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/runs/qwenimage21-kv-bench-20260929-111910/results/validation.json) · [Raw timings](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/runs/qwenimage21-kv-bench-20260929-111910/results/runs.csv) · [Environment and exact prompt](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/runs/qwenimage21-kv-bench-20260929-111910/results/metadata.json)

[HF Job](https://huggingface.co/jobs/sayakpaul/6abb9eb8c617607c354d4ea9) · [Run instructions](../../README.md) · [Benchmark script](results/benchmark.py) · [SVG](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/runs/qwenimage21-kv-bench-20260929-111910/qwenimage21_performance.svg) · [PDF](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/qwenimage21-benchmark/runs/qwenimage21-kv-bench-20260929-111910/qwenimage21_performance.pdf)
