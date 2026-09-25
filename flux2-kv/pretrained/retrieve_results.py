import json
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars


disable_progress_bars()
base = Path(__file__).parent
job = json.loads((base / "job.json").read_text())
volume = job["outputs"]
api = HfApi()
status = api.inspect_job(job_id=job["job_id"]).status
print(status)
api.sync_bucket(
    f"hf://buckets/{volume['source']}/{volume['path']}",
    str(base / "results"),
    quiet=True,
)
print(base / "results")
