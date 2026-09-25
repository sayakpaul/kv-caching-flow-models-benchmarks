# KV caching flow model benchmarks

- [FLUX.2 Klein: one reference image](flux2-kv/pretrained/README.md)
- [FLUX.2 Klein: three reference images](flux2-kv/pretrained_3refs/README.md)

Images, measurements, result JSON, charts, job records, and source archives are stored in the public [Hugging Face bucket](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks). Code, patches, and reports remain in Git. [artifacts.json](artifacts.json) lists every artifact's absolute download URL, size, and SHA-256 checksum, preserving the repository's directory layout in the bucket.

To restore all published artifacts into a fresh clone, run this from the repository root (Python 3.10 or newer; no additional packages or HF login needed):

```bash
python download_artifacts.py
```

To download only one benchmark's artifacts:

```bash
python download_artifacts.py --prefix flux2-kv/pretrained
python download_artifacts.py --prefix flux2-kv/pretrained_3refs
```

Run the full download before submitting jobs or comparing the two benchmarks. Downloaded files are ignored by Git. The downloader verifies every file and preserves existing files that differ from the published run unless `--overwrite` is supplied. Original result records and archives retain their recorded paths and contents; benchmark scripts use the restored local files.

Complete archive downloads: [one reference](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained/benchmark_artifacts.zip), [three references](https://huggingface.co/buckets/sayakpaul/kv-caching-flow-models-benchmarks/resolve/flux2-kv/pretrained_3refs/benchmark_3refs_artifacts.zip).
