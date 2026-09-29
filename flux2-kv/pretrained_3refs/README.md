# FLUX.2 Klein KV benchmark: three reference images

Artifacts are hosted in the public [HF bucket](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks). Before running the commands below, restore them with `hf buckets sync hf://buckets/sayakpaul/kv-caching-flow-models-benchmarks . --ignore-existing` from the repository root; see the [download instructions](../../README.md).

| Measurement | Without cache | With cache |
| --- | ---: | ---: |
| Median latency · 4 steps | 5.950 s | 3.446 s |
| Median latency · 8 steps | 11.459 s | 5.580 s |
| Peak GPU memory · both step counts | 35.45 GiB | 39.58 GiB |

Caching was **1.73× faster at 4 steps** and **2.05× faster at 8 steps**, using **4.12 GiB more peak memory**. Cached and uncached output images were pixel-identical at each step count. With [one reference](../pretrained/README.md), speedups were 1.30× and 1.40×.

Measured with `black-forest-labs/FLUX.2-klein-9b-kv` on an A100 80 GB in bfloat16: three 1024 × 704 references, 1024 × 1024 output, seed 42, **3 warmups and 5 measured runs per setting**. Latency covers the full pipeline call, excluding model loading and image saving. Memory is peak PyTorch allocated GPU memory, including weights. Results describe this input and setup.

## Apply the Diffusers patch

Before running locally, apply [`inputs/diffusers.patch`](inputs/diffusers.patch) to your Diffusers checkout and install it in your Python environment:

```bash
# From the benchmark repository root; set this to your Diffusers checkout.
diffusers_checkout=/path/to/diffusers
git -C "$diffusers_checkout" apply --check "$PWD/flux2-kv/pretrained_3refs/inputs/diffusers.patch"
git -C "$diffusers_checkout" apply "$PWD/flux2-kv/pretrained_3refs/inputs/diffusers.patch"
python -m pip install -e "$diffusers_checkout"
```

Apply it only once; both benchmarks use the same patch. It was tested against Diffusers commit `e0118ade2f60234c41bacf40330a7e2f61108849`.

The patch adds `use_kv_cache=False`, which recomputes reference tokens every step while preserving their fixed timestep and attention rules. This provides the uncached baseline; setting `kv_cache_mode=None` would change those rules.

## Run locally

Use a CUDA GPU with enough memory and authenticate with an HF account that has accepted the checkpoint license. This uses the same environment as the [one-reference benchmark](../pretrained/README.md).

```bash
# From the repository root.
cd flux2-kv/pretrained_3refs
python inputs/benchmark_flux2_klein_kv.py \
  --revision a6dfb36eca3a3906eb2fd460795adfb844e5fcce \
  --reference inputs/reference_01.png inputs/reference_02.png inputs/reference_03.png \
  --output results_local
```

Defaults run both cache modes at 4 and 8 steps, with the warmups and seed above. Every generated image, including warmups, is saved. Reference sources and preprocessing are recorded in [`inputs/reference_sources.json`](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/inputs/reference_sources.json).

## Run on HF Jobs

```bash
# From the repository root.
cd flux2-kv/pretrained_3refs
python submit_job.py
```

This launches a paid A100 job using [inputs/diffusers-source.tar.gz](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/inputs/diffusers-source.tar.gz), which already includes the patch. Local checkout edits do not change that bundle. Each submission creates a fresh directory under [runs/](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/tree/flux2-kv/pretrained_3refs/runs) and records the job in [job.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/job.json). Dependency versions come from the one-reference benchmark's metadata. After the job finishes, run `python retrieve_results.py` to download its outputs, then `python compare_results.py` to compare with the one-reference results.

## Results

[Completed HF Job](https://huggingface.co/jobs/sayakpaul/6ab513746b030d633f68e159). Its artifacts are in [`runs/flux2-kv-3refs-20260924-121120/`](runs/flux2-kv-3refs-20260924-121120/):

- [Results](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/tree/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/results): [performance chart](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/results/benchmark.png), all output PNGs, [measurements](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/results/runs.csv), [summary](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/results/summary.json), and [environment metadata](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/results/metadata.json).
- [Comparison](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/tree/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/comparison): [one-versus-three reference chart](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/comparison/reference_count_comparison.png), [output comparison](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/comparison/image_comparison.png), and [reference gallery](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/runs/flux2-kv-3refs-20260924-121120/comparison/references.png).
