# FLUX.2 Klein KV benchmark: one reference image

Artifacts are hosted in the public [HF bucket](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks).

| Measurement | Without cache | With cache |
| --- | ---: | ---: |
| Median latency · 4 steps | 3.085 s | 2.374 s |
| Median latency · 8 steps | 5.883 s | 4.209 s |
| Peak GPU memory · both step counts | 34.73 GiB | 35.48 GiB |

Caching was **1.30× faster at 4 steps** and **1.40× faster at 8 steps**, using **0.75 GiB more peak memory**. Cached and uncached output images were pixel-identical at each step count.

Measured with `black-forest-labs/FLUX.2-klein-9b-kv` on an A100 80 GB in bfloat16: one 1024 × 704 reference, 1024 × 1024 output, seed 42, **3 warmups and 5 measured runs per setting**. Latency covers the full pipeline call, excluding model loading and image saving. Memory is peak PyTorch allocated GPU memory, including weights. Results describe this input and setup.

## Apply the Diffusers patch

Before running locally, apply [`inputs/diffusers.patch`](inputs/diffusers.patch) to your Diffusers checkout and install it in your Python environment:

```bash
# From the benchmark repository root; set this to your Diffusers checkout.
diffusers_checkout=/path/to/diffusers
git -C "$diffusers_checkout" apply --check "$PWD/flux2-kv/pretrained/inputs/diffusers.patch"
git -C "$diffusers_checkout" apply "$PWD/flux2-kv/pretrained/inputs/diffusers.patch"
python -m pip install -e "$diffusers_checkout"
```

Apply it only once; both benchmarks use the same patch. It was tested against Diffusers commit `e0118ade2f60234c41bacf40330a7e2f61108849`.

The patch adds `use_kv_cache=False`, which recomputes reference tokens every step while preserving their fixed timestep and attention rules. This provides the uncached baseline; setting `kv_cache_mode=None` would change those rules.

## Run locally

Use a CUDA GPU with enough memory and authenticate with an HF account that has accepted the checkpoint license. The tested environment uses PyTorch 2.10.0 (CUDA 12.8), Transformers 5.14.1, and Accelerate 1.13.0; full dependency versions are in [`results/metadata.json`](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/metadata.json).

```bash
# From the repository root.
cd flux2-kv/pretrained
python inputs/benchmark_flux2_klein_kv.py \
  --revision a6dfb36eca3a3906eb2fd460795adfb844e5fcce \
  --reference inputs/reference.png \
  --output results_local
```

Defaults run both cache modes at 4 and 8 steps, with the warmups and seed above. Every generated image, including warmups, is saved.

## Run on HF Jobs

```bash
# From the repository root.
cd flux2-kv/pretrained
python submit_job.py
```

This launches a paid A100 job using [inputs/diffusers-source.tar.gz](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/inputs/diffusers-source.tar.gz), which already includes the patch. Local checkout edits do not change that bundle. The launcher reuses its output paths; change them in `submit_job.py` before rerunning if you want to preserve the previous results. After the job finishes, run `python retrieve_results.py` to download its outputs.

## Results

[Completed HF Job](https://huggingface.co/jobs/sayakpaul/6ab50f3a6b030d633f68e080)

- [Performance chart](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/benchmark.png) and [output comparison](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/image_comparison.png).
- [Measurements](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/runs.csv), [summary](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/summary.json), and [environment](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/results/metadata.json).
- All output PNGs: [results directory](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/tree/flux2-kv/pretrained/results).
