import json
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import disable_progress_bars


base = Path(__file__).parent
disable_progress_bars()
api = HfApi()
inputs = api.sync_job_volume(base / "inputs", "/inputs", remote_name="flux2-kv-benchmark-inputs-v2")
outputs = api.sync_job_volume(base / "results", "/outputs", remote_name="flux2-kv-benchmark-results-v2", read_only=False)
dependencies = [
    "torch==2.10.0",
    "transformers==5.14.1",
    "accelerate==1.13.0",
    "huggingface-hub==1.33.0",
    "safetensors==0.8.0",
    "numpy",
    "Pillow",
    "matplotlib",
    "scipy",
    "sentencepiece",
    "protobuf",
    "ftfy",
    "regex",
    "requests",
    "importlib_metadata",
    "filelock",
]
job = api.run_uv_job(
    str(base / "run_job.py"),
    script_args=[
        "--revision", "a6dfb36eca3a3906eb2fd460795adfb844e5fcce",
        "--reference", "/inputs/reference.png",
        "--output", "/outputs",
    ],
    dependencies=dependencies,
    python="3.12",
    flavor="a100-large",
    timeout="45m",
    name="flux2-klein-kv-benchmark",
    secrets={"HF_TOKEN": get_token()},
    env={"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"},
    volumes=[inputs, outputs],
)
record = {
    "job_id": job.id,
    "job_url": job.url,
    "flavor": "a100-large",
    "dependencies": dependencies,
    "inputs": asdict(inputs),
    "outputs": asdict(outputs),
}
(base / "job.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
