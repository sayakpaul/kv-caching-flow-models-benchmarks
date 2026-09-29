# KV caching flow model benchmarks

- [FLUX.2 Klein: one reference image](flux2-kv/pretrained/README.md)
- [FLUX.2 Klein: three reference images](flux2-kv/pretrained_3refs/README.md)
- [Reference KV caching + TaylorSeer](taylorseer_benchmark/README.md)

Images, saved latents, measurements, result JSON, charts, job records, logs, and source archives are stored in the public [Hugging Face bucket](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks), preserving the repository's directory layout. Code, patches, and reports remain in Git.

To restore all published artifacts into a fresh clone, run this from the repository root using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/buckets). The bucket is public; downloads do not require an HF login.

```bash
hf buckets sync hf://buckets/sayakpaul/kv-caching-flow-models-benchmarks . --ignore-existing
```

To download only one benchmark's artifacts:

```bash
hf buckets sync hf://buckets/sayakpaul/kv-caching-flow-models-benchmarks/flux2-kv/pretrained flux2-kv/pretrained --ignore-existing
hf buckets sync hf://buckets/sayakpaul/kv-caching-flow-models-benchmarks/flux2-kv/pretrained_3refs flux2-kv/pretrained_3refs --ignore-existing
hf buckets sync hf://buckets/sayakpaul/kv-caching-flow-models-benchmarks/taylorseer_benchmark taylorseer_benchmark --ignore-existing
```

Run the full download before submitting jobs or comparing benchmarks. Downloaded files are ignored by Git, and `--ignore-existing` preserves existing local files. Original result records and archives retain their recorded paths and contents; benchmark scripts use the restored local files.

Complete archive downloads: [one reference](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/benchmark_artifacts.zip), [three references](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/benchmark_3refs_artifacts.zip).
