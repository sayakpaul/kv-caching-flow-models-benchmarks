import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import disable_progress_bars


base = Path(__file__).parent
disable_progress_bars()
api = HfApi()
run_name = "qwenimage21-kv-bench-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
run_directory = base / "runs" / run_name
output_directory = run_directory / "results"
output_directory.mkdir(parents=True)
inputs = api.sync_job_volume(base / "inputs", "/inputs", remote_name=run_name + "-inputs")
outputs = api.sync_job_volume(output_directory, "/outputs", remote_name=run_name + "-results", read_only=False)
baseline = json.loads((base.parent / "flux2-kv/pretrained/results/metadata.json").read_text())
dependencies = [
    f"{name}=={version.split('+')[0] if name.lower() == 'torch' else version}"
    for name, version in sorted(baseline["packages"].items())
]
dependencies = [x for x in dependencies if not x.lower().startswith(("transformers==", "tokenizers=="))]
dependencies += ["transformers==5.17.0", "tokenizers==0.23.1", "torchvision==0.25.0"]
job = api.run_uv_job(
    str(base / "run_job.py"),
    script_args=[
        "--reference",
        "/inputs/reference.png",
        "--output",
        "/outputs",
    ],
    dependencies=dependencies,
    python="3.12",
    flavor="a100-large",
    timeout="90m",
    name=run_name,
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
    "local_output_directory": str(output_directory.relative_to(base)),
}
for path in (base / "job.json", run_directory / "job.json"):
    path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: v for k, v in record.items() if k != "dependencies"}, indent=2))
