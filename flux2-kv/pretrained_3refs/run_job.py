import runpy
import shutil
import sys
import tarfile
from pathlib import Path


source = Path("/tmp/flux2-kv-source")
source.mkdir(parents=True, exist_ok=True)
with tarfile.open("/inputs/diffusers-source.tar.gz") as archive:
    archive.extractall(source, filter="data")
sys.path.insert(0, str(source / "src"))
output = Path("/outputs")
output.mkdir(parents=True, exist_ok=True)
for name in ("source_manifest.json", "diffusers.patch", "benchmark_flux2_klein_kv.py", "validate_patch.py", "reference_sources.json"):
    shutil.copy2(Path("/inputs") / name, output / name)
sys.argv = [str(source / "benchmarks/benchmark_flux2_klein_kv.py"), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
