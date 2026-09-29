import runpy
import shutil
import sys
import tarfile
from pathlib import Path


source = Path("/tmp/flux2-taylorseer-source")
source.mkdir(parents=True, exist_ok=True)
with tarfile.open("/inputs/diffusers-source.tar.gz") as archive:
    archive.extractall(source, filter="data")
sys.path.insert(0, str(source / "src"))
sys.path.insert(0, "/inputs")
output = Path("/outputs")
output.mkdir(parents=True, exist_ok=True)
for name in ("config.py", "benchmark.py", "validate_taylorseer.py", "source_info.json"):
    shutil.copy2(Path("/inputs") / name, output / name)
runpy.run_path("/inputs/validate_taylorseer.py", run_name="__main__")
sys.argv = ["/inputs/benchmark.py", *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
