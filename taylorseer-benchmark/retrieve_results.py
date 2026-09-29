import json
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars


disable_progress_bars()
base = Path(__file__).parent
job = json.loads((base / "job.json").read_text())
volume = job["outputs"]
output_directory = Path(job["local_output_directory"])
if output_directory.is_absolute():
    # Archived jobs recorded the original checkout location.
    output_directory = base / "runs" / output_directory.parent.name / output_directory.name
else:
    output_directory = base / output_directory
api = HfApi()
print(api.inspect_job(job_id=job["job_id"]).status)
api.sync_bucket(
    f"hf://buckets/{volume['source']}/{volume['path']}",
    str(output_directory),
    quiet=True,
)
print(output_directory)
