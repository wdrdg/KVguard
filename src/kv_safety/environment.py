"""Record the execution host: OS, Python, GPU, driver, CUDA, RAM, disk, package versions.

python -m kv_safety.environment [--config CFG] [--out PATH]
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .config import REPO_ROOT, artifacts_dir, load_config, write_json

PACKAGES = [
    "torch", "transformers", "accelerate", "huggingface_hub", "safetensors", "tokenizers",
    "numpy", "scikit-learn", "pandas", "pyarrow", "PyYAML", "pytest", "tqdm", "matplotlib", "datasets",
]


def _mem_total_gib() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 2)
    except OSError:
        pass
    return None


def _nvidia_smi() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def collect(disk_path: Path | None = None) -> dict:
    info: dict = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "ram_total_gib": _mem_total_gib(),
        "nvidia_smi": _nvidia_smi(),
    }
    disk_path = disk_path or REPO_ROOT
    du = shutil.disk_usage(disk_path)
    info["disk"] = {"path": str(disk_path), "free_gib": round(du.free / 1024**3, 2),
                    "total_gib": round(du.total / 1024**3, 2)}
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    info["hf_home"] = hf_home
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    info["packages"] = versions
    try:
        import torch

        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpus"] = [
                {"index": i, "name": torch.cuda.get_device_name(i),
                 "total_mib": torch.cuda.get_device_properties(i).total_memory // 2**20,
                 "capability": ".".join(map(str, torch.cuda.get_device_capability(i)))}
                for i in range(torch.cuda.device_count())
            ]
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
            info["cudnn"] = torch.backends.cudnn.version()
        else:
            info["gpus"] = []
    except ImportError:
        info["cuda_available"] = None
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    info = collect()
    if args.out:
        out = Path(args.out)
    elif args.config:
        out = artifacts_dir(load_config(args.config)) / "environment.json"
    else:
        out = REPO_ROOT / "artifacts" / "environment.json"
    write_json(out, info)
    print(json.dumps(info, indent=2))
    if not info.get("cuda_available"):
        print("\nWARNING: no CUDA device visible. GPU alignment tests and CUDA benchmarks cannot run here.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
