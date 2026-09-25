import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


root = Path("/home/sayakpaul/diffusers")
base = Path(__file__).parent
inputs = base / "inputs"
files = subprocess.check_output(
    ["git", "ls-files", "src", "setup.py", "pyproject.toml", "README.md", "LICENSE"], cwd=root, text=True
).splitlines()
files.append("benchmarks/benchmark_flux2_klein_kv.py")
with tarfile.open(inputs / "diffusers-source.tar.gz", "w:gz") as archive:
    for name in files:
        archive.add(root / name, arcname=name)
(inputs / "diffusers.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=root))
shutil.copy2(root / "benchmarks/benchmark_flux2_klein_kv.py", inputs / "benchmark_flux2_klein_kv.py")
shutil.copy2(base / "validate_patch.py", inputs / "validate_patch.py")
manifest = {
    "diffusers_base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "source_archive_sha256": hashlib.sha256((inputs / "diffusers-source.tar.gz").read_bytes()).hexdigest(),
    "patch_sha256": hashlib.sha256((inputs / "diffusers.patch").read_bytes()).hexdigest(),
    "model_revision": "a6dfb36eca3a3906eb2fd460795adfb844e5fcce",
    "reference_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
    "source_sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "src/diffusers/models/transformers/transformer_flux2.py",
            "src/diffusers/pipelines/flux2/pipeline_flux2_klein_kv.py",
            "benchmarks/benchmark_flux2_klein_kv.py",
        )
    },
    "validation": "Tiny CPU float32 outputs exactly equal for 0, 1, and 2 reference images; uncached execution fails the test if any Flux2KVCache is allocated.",
}
(inputs / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Source archive bytes:", (inputs / "diffusers-source.tar.gz").stat().st_size)
