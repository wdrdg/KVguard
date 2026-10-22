"""Load the frozen backbone/tokenizer at the pinned revision with the fixed settings."""

from __future__ import annotations

import torch

from .config import apply_offline, configured_layers, pinned_revision, read_manifest
from .features.cache_reader import check_model_cache_compat

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _source(cfg: dict) -> tuple[str, str | None, bool]:
    apply_offline(cfg)
    rev = pinned_revision(cfg, "model")
    offline = bool(cfg.get("offline"))
    if offline and rev is None:
        raise SystemExit("offline mode needs a pinned model revision; run kv_safety.download first")
    return cfg["model"]["id"], rev, offline


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    repo, rev, offline = _source(cfg)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=cfg.get("hf_cache_dir"),
                                        local_files_only=offline)
    if not tok.is_fast:
        raise RuntimeError("a fast tokenizer (offset mapping) is required")
    return tok


def assert_expected_config(model_config, expected: dict) -> None:
    got = {k: getattr(model_config, k) for k in expected}
    if got != expected:
        raise RuntimeError(f"model config {got} != expected {expected}; wrong checkpoint variant?")


def load_model(cfg: dict, device: str | torch.device):
    from transformers import AutoModelForCausalLM

    repo, rev, offline = _source(cfg)
    mc = cfg["model"]
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=rev, cache_dir=cfg.get("hf_cache_dir"), local_files_only=offline,
        dtype=DTYPES[mc["dtype"]], attn_implementation=mc["attn_implementation"],
    ).to(device)
    prepare_frozen(model, cfg)
    return model


def prepare_frozen(model, cfg: dict) -> None:
    mc = cfg["model"]
    if mc.get("expected_config"):
        assert_expected_config(model.config, mc["expected_config"])
    check_model_cache_compat(model.config)
    impl = getattr(model.config, "_attn_implementation", None)
    if impl != mc["attn_implementation"]:
        raise RuntimeError(f"attention backend {impl} != configured {mc['attn_implementation']}")
    n_layers = model.config.num_hidden_layers
    for layer in configured_layers(cfg):
        if not (0 <= layer < n_layers):
            raise ValueError(f"layer {layer} out of range for num_hidden_layers={n_layers}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def model_provenance(cfg: dict) -> dict:
    m = read_manifest(cfg).get("model", {})
    return {"model_id": cfg["model"]["id"], "revision": pinned_revision(cfg, "model"),
            "chat_template_sha256": m.get("offline_check", {}).get("chat_template_sha256")}
