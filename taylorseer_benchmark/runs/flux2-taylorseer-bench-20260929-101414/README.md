# Reference KV caching + TaylorSeer

| Measurement | Reference KV cache | + TaylorSeer |
| --- | ---: | ---: |
| Latency (4 steps) | 2.407s | 2.033s (15.5% lower) |
| Latency (8 steps) | 4.264s | 3.105s (27.2% lower) |
| Peak GPU memory | 35.48 GiB | 37.24 GiB |

A100-SXM4-80GB, bfloat16, 1024 × 1024 output, one cat reference, seed 42. Each setting has three warmups followed by five timed calls, with alternating mode order. The baseline is measured again in this job. Text KV caching is disabled in both cases.

Latency is the median of five complete pipeline calls: prompt/reference encoding, cache construction, denoising, decoding, PIL conversion, and cache reset. It excludes model loading, saving PNGs, garbage collection, and installing/removing TaylorSeer hooks. Peak memory is the largest peak allocated GPU memory across those five calls, including weights and caches. No offload, compilation, or quantization.

![Performance](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/taylorseer_performance.png)

## Configuration

TaylorSeer uses the built-in default attention-module hooks in all 8 double and 24 single blocks, with `cache_interval=2`, `disable_cache_before_step=3`, `max_order=1`, and bfloat16 factors. Single-stream attention modules also include the parallel MLP computation. The first three steps compute fully. At four steps, step 4 is predicted; at eight steps, steps 4, 6, and 8 are predicted. All other steps compute and update the factors.

The three initial full steps allow the attention output shapes to stabilize after reference tokens leave the live sequence. The pinned implementation restarts Taylor factors on a shape change. Each pipeline call runs inside `transformer.cache_context("cond")`; the pipeline clears stateful hooks at the end. No Diffusers patch is needed at this revision.

```python
from diffusers import TaylorSeerCacheConfig

pipe.transformer.enable_cache(TaylorSeerCacheConfig(
    cache_interval=2,
    disable_cache_before_step=3,
    max_order=1,
    taylor_factors_dtype=torch.bfloat16,
))
with pipe.transformer.cache_context("cond"):
    image = pipe(image=reference, prompt=prompt, num_inference_steps=8,
                 generator=torch.Generator("cuda").manual_seed(42)).images[0]
```

## Image comparisons and validation

TaylorSeer is approximate and changes the generated images. These parameters were chosen before viewing results; this is one prompt/reference pair, not a general quality benchmark. All first-step latents match the baseline. Audits confirm all 32 reference caches are populated and all TaylorSeer hooks follow the stated schedule. Tiny-model checks cover shape changes, exact no-prediction controls, and repeatability across calls.

Visual inspection: the wizard-cat composition remains recognizable in all six comparisons. At four steps, all three seeds show noticeably rougher, oversharpened-looking fur and fabric, with stronger background texture. At eight steps the differences are less pronounced, but texture and contrast still change. The measured speedup is therefore not quality-neutral.

![Seed 42](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/comparison_seed_42.png)

[Seed 43](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/comparison_seed_43.png) · [Seed 44](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/comparison_seed_44.png) · [Pixel differences](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/results/quality_metrics.json)

All 32 warmup/timed PNGs plus 12 quality PNGs are saved and their hashes verified. Final quality latents are saved alongside each image. Every timed/warmup output matches its seed-42 quality image exactly.

[HF Job](https://huggingface.co/jobs/sayakpaul/6abb8f80e2f3c356be038fab) · [Launch instructions](../../README.md) · [Measurement script](results/benchmark.py) · [Configuration and audit](results/config.py)

[Raw runs](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/results/runs.csv) · [Summary](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/results/summary.json) · [Environment and prompt](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/results/metadata.json) · [Validation](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/results/validation.json) · [SVG](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/taylorseer_performance.svg) · [PDF](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/taylorseer_performance.pdf)
