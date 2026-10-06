# KV caching flow model benchmarks

Accompanying benchmarking codebase for the post: [KV Caching in Flow Models](https://sayak.dev/kv-flow).

Refer to the respective directories for instructions on how to run the code.

- [FLUX.2 Klein: one reference image](flux2-kv/pretrained/README.md)
- [FLUX.2 Klein: three reference images](flux2-kv/pretrained_3refs/README.md)
- [Reference KV caching + TaylorSeer](taylorseer-benchmark/README.md)
- [Qwen-Image-2.1: KV caching](qwenimage21-benchmark/README.md)

Images, saved latents, measurements, result JSON, charts, job records, logs, and source archives are stored in the public [Hugging Face bucket](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks).

## AI assistance

The benchmarking was done with assistance from Codex. I reviewed the benchmarking code and the patches manually
to ensure I knew what I was doing.
