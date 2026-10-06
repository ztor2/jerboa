from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path


BASE_URL = "https://api.aihub.or.kr"
API_VERSION = "0.6"


def load_env_api_key() -> str:
    key = os.getenv("AIHUB_API_KEY")
    if key:
        return key.strip().strip("'\"")

    env_paths = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]

    for p in env_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("AIHUB_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip("'\"")
                        if val:
                            return val
    return ""


def _run_curl(url: str, headers: dict[str, str] | None = None, output_file: Path | None = None) -> tuple[int, str]:
    cmd = ["curl", "-sSL"]
    if headers:
        for k, v in headers.items():
            cmd.extend(["-H", f"{k}: {v}"])
    if output_file:
        cmd.extend(["-o", str(output_file)])
    cmd.append(url)

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout


def list_datasets(query: str | None = None) -> None:
    url = f"{BASE_URL}/info/dataset.do"
    ret, content = _run_curl(url)
    if ret != 0:
        print("Failed to fetch dataset list from AI-Hub.", file=sys.stderr)
        return

    lines = content.splitlines()
    for line in lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("==="):
            continue
        if query:
            if re.search(query, line_clean, re.IGNORECASE):
                print(line_clean)
        else:
            print(line_clean)


def list_files(dataset_key: str | int) -> None:
    url = f"{BASE_URL}/info/{dataset_key}.do"
    ret, content = _run_curl(url)
    if ret != 0:
        print(f"Failed to fetch files for dataset {dataset_key}.", file=sys.stderr)
        return
    print(content)


def _merge_parts(target_dir: Path) -> None:
    part_files = list(target_dir.glob("*.part*"))
    if not part_files:
        return

    prefixes = set()
    for pf in part_files:
        m = re.match(r"^(.*)\.part[0-9]+$", pf.name)
        if m:
            prefixes.add(m.group(1))

    for prefix in sorted(prefixes):
        matched = sorted(
            target_dir.glob(f"{prefix}.part*"),
            key=lambda p: int(p.name.split(".part")[-1]) if p.name.split(".part")[-1].isdigit() else p.name
        )
        dest_file = target_dir / prefix
        print(f"Merging {len(matched)} parts -> {dest_file.name}")
        with open(dest_file, "wb") as outfile:
            for part in matched:
                with open(part, "rb") as infile:
                    while chunk := infile.read(1024 * 1024 * 4):
                        outfile.write(chunk)
                part.unlink()


def download_dataset(
    dataset_key: str | int,
    file_keys: str | None = None,
    dest_dir: str | Path = "data/raw/aihub",
    api_key: str | None = None,
) -> None:
    resolved_key = api_key or load_env_api_key()
    if not resolved_key:
        print("Error: AIHUB_API_KEY not found in environment or .env file.", file=sys.stderr)
        sys.exit(1)

    file_sn = file_keys if file_keys else "all"
    download_url = f"{BASE_URL}/down/{API_VERSION}/{dataset_key}.do?fileSn={file_sn}"

    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    tar_path = dest_path / "download.tar"

    headers = {
        "apikey": resolved_key,
    }

    print(f"Connecting to AI-Hub for dataset {dataset_key} (files: {file_sn})...")
    ret, _ = _run_curl(download_url, headers=headers, output_file=tar_path)
    if ret != 0 or not tar_path.exists():
        print(f"Curl failed with return code {ret}", file=sys.stderr)
        sys.exit(1)

    # Check for error responses
    if tar_path.stat().st_size < 1024:
        try:
            with open(tar_path, "r", encoding="utf-8") as f:
                content = f.read()
                if "Error" in content or "제한" in content or "승인" in content or "failed" in content:
                    print(f"Server message: {content.strip()}", file=sys.stderr)
                    tar_path.unlink()
                    sys.exit(1)
        except UnicodeDecodeError:
            pass

    print(f"Extracting {tar_path.name} into {dest_path}...")
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=dest_path)
    except Exception as e:
        print(f"Failed to extract tar archive: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if tar_path.exists():
            tar_path.unlink()

    for root, _, _ in os.walk(dest_path):
        _merge_parts(Path(root))

    print(f"Download and extraction completed at: {dest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Hub dataset exploration & download CLI for Jerboa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_p = subparsers.add_parser("list", help="List available datasets")
    list_p.add_argument("-q", "--query", help="Filter datasets with regex/keyword")

    # files
    files_p = subparsers.add_parser("files", help="List files in a dataset")
    files_p.add_argument("dataset_key", help="Target dataset key (e.g. 71633)")

    # download
    down_p = subparsers.add_parser("download", help="Download dataset or specific files")
    down_p.add_argument("dataset_key", help="Target dataset key (e.g. 71633)")
    down_p.add_argument("-f", "--files", help="Comma-separated file keys (default: all)")
    down_p.add_argument("-o", "--output", default="data/raw/aihub", help="Output directory")
    down_p.add_argument("-k", "--api-key", help="AI-Hub API Key (defaults to .env)")

    args = parser.parse_args()

    if args.command == "list":
        list_datasets(args.query)
    elif args.command == "files":
        list_files(args.dataset_key)
    elif args.command == "download":
        download_dataset(args.dataset_key, args.files, args.output, args.api_key)


if __name__ == "__main__":
    main()
