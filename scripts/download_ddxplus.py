#!/usr/bin/env python3
"""Download the official English DDXPlus release into the local project data dir."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import urllib.request


FIGSHARE_ARTICLE_ID = 22687585
DEFAULT_DATASET_DIR = Path(os.environ.get("DDXPLUS_DATASET_DIR", "dataset"))
MANIFEST_FILE = "manifest.json"


@dataclass
class DatasetFile:
    name: str
    size: int
    download_url: str
    supplied_md5: str | None = None


def fetch_figshare_manifest(article_id: int = FIGSHARE_ARTICLE_ID) -> list[DatasetFile]:
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    with urllib.request.urlopen(api_url) as response:
        payload = json.load(response)
    return [
        DatasetFile(
            name=item["name"],
            size=item["size"],
            download_url=item["download_url"],
            supplied_md5=item.get("supplied_md5"),
        )
        for item in payload["files"]
    ]


def md5_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(destination.suffix + ".part")
    if temp_destination.exists():
        temp_destination.unlink()
    with urllib.request.urlopen(url) as response, temp_destination.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
    temp_destination.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the DDXPlus English dataset from Figshare.")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Target directory for the dataset files.",
    )
    parser.add_argument(
        "--dataset-dir",
        dest="output_dir",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Fetch the manifest only without downloading the files.",
    )
    parser.add_argument(
        "--no-hash-check",
        action="store_true",
        help="Skip MD5 validation after download.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir or DEFAULT_DATASET_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = fetch_figshare_manifest()
    (output_dir / MANIFEST_FILE).write_text(
        json.dumps([asdict(item) for item in manifest], indent=2),
        encoding="utf-8",
    )

    if not args.metadata_only:
        for item in manifest:
            destination = output_dir / item.name
            if destination.exists() and item.supplied_md5 and not args.no_hash_check:
                if md5_of_file(destination) == item.supplied_md5:
                    continue
                destination.unlink()
            elif destination.exists() and args.no_hash_check:
                continue

            print(f"Downloading {item.name}...")
            download_file(item.download_url, destination)
            if item.supplied_md5 and not args.no_hash_check:
                observed = md5_of_file(destination)
                if observed != item.supplied_md5:
                    raise ValueError(f"MD5 mismatch for {item.name}: {observed} != {item.supplied_md5}")

    print(f"Target directory: {output_dir}")
    for item in manifest:
        print(f"- {item.name} ({item.size} bytes)")


if __name__ == "__main__":
    main()
