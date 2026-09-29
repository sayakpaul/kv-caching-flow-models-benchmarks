# Reference KV caching + TaylorSeer

[Results](runs/flux2-taylorseer-bench-20260929-101414/README.md) · [Performance chart](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/taylorseer_performance.png) · [Image comparison](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/runs/flux2-taylorseer-bench-20260929-101414/comparison_seed_42.png)

All images, saved latents, result data, logs, and the source archive are in the public [HF bucket](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/tree/taylorseer_benchmark).

This compares the normal `Flux2KleinKVPipeline` reference-image KV cache against that same cache plus [TaylorSeer](https://huggingface.co/docs/diffusers/main/en/optimization/cache#taylorseer-cache). It does not enable text KV caching.

The benchmark uses the pinned FLUX.2 Klein 9B KV checkpoint, one cat reference, the same wizard-edit prompt, and seed 42. It measures four and eight steps on an A100 80 GB, with three warmups and five timed calls per setting. Latency covers the complete pipeline; peak allocated GPU memory includes weights and caches. Both cases are measured in the same job with alternating order. Images are saved for every call, plus separate comparisons for seeds 42, 43, and 44.

## Run with HF Jobs

```bash
# From the repository root.
cd taylorseer_benchmark
python submit_job.py
```

This starts a paid A100 job using pinned dependencies and unique input/output paths. Accept the gated model license and log in with a Hugging Face token that has model and Jobs access.

After completion:

```bash
python retrieve_results.py
python make_report.py
```

[job.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/job.json) identifies the latest run. The report script verifies image hashes and creates the compact table, PNG/SVG/PDF performance charts, and three image grids without a reference thumbnail.

## Source and configuration

The job uses an unmodified archive of Diffusers commit `5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2`, stored in [inputs/diffusers-source.tar.gz](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/inputs/diffusers-source.tar.gz). **No patch is required for this experiment.** The older [Diffusers patch](../flux2-kv/pretrained/inputs/diffusers.patch) enables the faithful no-reference-cache baseline used in previous experiments; do not apply it here.

TaylorSeer is configured with `cache_interval=2`, `disable_cache_before_step=3`, `max_order=1`, and bfloat16 factors. The default attention-module hooks cover all double/single blocks. With one-based step numbers, step 4 is predicted in the four-step run; steps 4, 6, and 8 are predicted in the eight-step run. All remaining steps compute normally.

Three initial full steps handle the reference-token shape transition and provide two observations at the later shape. The pinned implementation resets factors when shapes change. `transformer.cache_context("cond")` supplies the hook context, and the pipeline resets cache state at the end of every call.

- `inputs/config.py`: exact configuration and audit helpers.
- `inputs/validate_taylorseer.py`: tiny-model checks before checkpoint loading.
- `inputs/benchmark.py`: quality checks, warmups, timing, memory, and serialization.
- `run_job.py`: remote entry point.
- [source_manifest.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/source_manifest.json): input hashes.

For local reproduction, install that Diffusers revision in a separate checkout and use the dependency versions recorded in [job.json](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/taylorseer_benchmark/job.json). Then run:

```bash
python inputs/validate_taylorseer.py
python inputs/benchmark.py --reference inputs/reference.png --output local-results
```

TaylorSeer is an approximation and changes outputs. This experiment measures one prompt/reference pair and does not establish general quality preservation.
