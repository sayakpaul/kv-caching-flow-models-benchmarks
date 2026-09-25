"""Restore published benchmark artifacts using the URLs and hashes in artifacts.json."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default="",
        help="Only restore paths under this repository-relative prefix",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT,
        help="Destination directory (default: repository root)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace local files that differ from the published run",
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "artifacts.json").read_text())
    prefix = args.prefix.rstrip("/")
    entries = [
        entry
        for entry in manifest["files"]
        if not prefix
        or entry["path"] == prefix
        or entry["path"].startswith(prefix + "/")
    ]
    if not entries:
        parser.error(f"No artifacts match {args.prefix!r}")
    destination = args.destination.resolve()
    for entry in entries:
        target = (destination / entry["path"]).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Artifact path escapes destination: {entry['path']}")
        if target.exists():
            if (
                target.stat().st_size == entry["size"]
                and sha256(target) == entry["sha256"]
            ):
                continue
            if not args.overwrite:
                raise FileExistsError(
                    f"Local artifact differs: {target}. Use --overwrite to restore the published file."
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                with urlopen(entry["url"], timeout=120) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(chunk)
            if (
                temporary.stat().st_size != entry["size"]
                or sha256(temporary) != entry["sha256"]
            ):
                raise ValueError(
                    f"Downloaded artifact failed integrity check: {entry['url']}"
                )
            temporary.replace(target)
            print(entry["path"], flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"Verified {len(entries)} artifacts in {destination}")


if __name__ == "__main__":
    main()
