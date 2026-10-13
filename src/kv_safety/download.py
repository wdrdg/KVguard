"""Download the backbone and datasets at pinned revisions and write resource_manifest.json.

python -m kv_safety.download --config configs/pilot.yaml [--skip-model] [--verify-weights]

Every resource is resolved to an immutable commit SHA; that SHA is what later stages load
(config `revision: null` -> manifest SHA). Downloads resume via huggingface_hub. Nothing
is silently substituted: a failure is reported and the script exits non-zero.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import apply_offline, base_parser, load_config, manifest_path, read_manifest, write_json

MODEL_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE*", "README.md", "*.jinja"]
MIN_FREE_GIB_MODEL = 20.0


def sha256_file(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def _files_record(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out.append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                        "sha256": sha256_file(p.resolve())})
    return out


def _resolve_sha(api, repo_id: str, repo_type: str, revision: str | None) -> tuple[str, dict]:
    if repo_type == "model":
        info = api.model_info(repo_id, revision=revision)
    else:
        info = api.dataset_info(repo_id, revision=revision)
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    license_ = None
    if card is not None:
        license_ = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
    return info.sha, {"license": license_}


def _check_disk(path: Path, need_gib: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free / 1024**3
    if free < need_gib:
        raise SystemExit(f"Not enough free disk at {path}: {free:.1f} GiB < {need_gib} GiB required")


def download_model(cfg: dict, api, verify_weights: bool) -> dict:
    from huggingface_hub import snapshot_download

    repo = cfg["model"]["id"]
    sha, meta = _resolve_sha(api, repo, "model", cfg["model"].get("revision"))
    cache_dir = cfg.get("hf_cache_dir")
    _check_disk(Path(cache_dir or os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")),
                MIN_FREE_GIB_MODEL)
    local = Path(snapshot_download(repo, revision=sha, allow_patterns=MODEL_PATTERNS, cache_dir=cache_dir))
    rec = {"repo_id": repo, "repo_type": "model", "revision": sha, "local_path": str(local), **meta,
           "files": _files_record(local)}
    rec["offline_check"] = verify_model_offline(local, cfg, verify_weights)
    return rec


def verify_model_offline(local: Path, cfg: dict, verify_weights: bool) -> dict:
    from transformers import AutoConfig, AutoTokenizer

    conf = AutoConfig.from_pretrained(local, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(local, local_files_only=True)
    exp = cfg["model"]["expected_config"]
    got = {k: getattr(conf, k) for k in exp}
    if got != exp:
        raise SystemExit(f"Downloaded config {got} != expected {exp}; wrong variant? Not substituting.")
    tmpl = tok.chat_template or ""
    res = {"config": got, "tokenizer_class": type(tok).__name__, "is_fast": tok.is_fast,
           "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(),
           "use_sliding_window": getattr(conf, "use_sliding_window", None),
           "weights_loaded": False}
    if verify_weights:
        import torch
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(local, local_files_only=True, torch_dtype=torch.bfloat16)
        res["weights_loaded"] = True
        res["n_parameters"] = sum(p.numel() for p in m.parameters())
        del m
    return res


def download_dataset_files(cfg: dict, api, key: str, files: list[str] | None) -> dict:
    from huggingface_hub import hf_hub_download, snapshot_download

    dcfg = cfg["data"][key]
    repo = dcfg["repo_id"]
    sha, meta = _resolve_sha(api, repo, "dataset", dcfg.get("revision"))
    cache_dir = cfg.get("hf_cache_dir")
    if files:
        paths = [Path(hf_hub_download(repo, f, repo_type="dataset", revision=sha, cache_dir=cache_dir))
                 for f in files + ["README.md"]]
        root = paths[0].parent
    else:
        root = Path(snapshot_download(repo, repo_type="dataset", revision=sha, cache_dir=cache_dir))
    return {"repo_id": repo, "repo_type": "dataset", "revision": sha, "local_path": str(root), **meta,
            "files": _files_record(root)}


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-qwen3guardtest", action="store_true")
    ap.add_argument("--verify-weights", action="store_true",
                    help="also load the full BF16 weights on CPU (~8 GiB RAM) to prove offline loading")
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_offline(cfg)
    from huggingface_hub import HfApi

    api = HfApi()
    manifest = read_manifest(cfg)
    failures = {}
    steps = [("fineharm", lambda: download_dataset_files(cfg, api, "fineharm",
                                                         list(cfg["data"]["fineharm"]["files"].values())))]
    if not args.skip_qwen3guardtest:
        steps.append(("qwen3guardtest", lambda: download_dataset_files(cfg, api, "qwen3guardtest", None)))
    if not args.skip_model:
        steps.append(("model", lambda: download_model(cfg, api, args.verify_weights)))
    for key, fn in steps:
        print(f"[download] {key} ...", flush=True)
        try:
            rec = fn()
            rec["downloaded_at"] = datetime.now(timezone.utc).isoformat()
            prev = manifest.get(key)
            if prev and prev.get("revision") != rec["revision"]:
                print(f"[download] NOTE {key}: revision changed {prev['revision']} -> {rec['revision']}")
            manifest[key] = rec
            print(f"[download] {key} OK revision={rec['revision']} files={len(rec['files'])}")
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            failures[key] = f"{type(e).__name__}: {e}"
            print(f"[download] {key} FAILED: {failures[key]}", file=sys.stderr)
    write_json(manifest_path(cfg), manifest)
    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
