from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def ensure_merged_if_lora_adapter(model_id: str, hf_dtype: str = "bfloat16") -> str:
    """
    If `model_id` points to a PEFT LoRA adapter directory (adapter_model.safetensors + adapter_config.json),
    merge it into a standalone HF model directory and return the merged directory path.

    Otherwise, return `model_id` unchanged.

    Cache key is derived from (absolute adapter path + adapter_model mtime), so repeated runs reuse the merge.
    You can override the cache root with env var: AUTOMINGO_MERGED_DIR
    """
    p = Path(model_id)
    if not p.exists() or not p.is_dir():
        return model_id

    adapter_weights = p / "adapter_model.safetensors"
    adapter_cfg = p / "adapter_config.json"
    if not adapter_weights.exists() or not adapter_cfg.exists():
        return model_id

    cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
    base = cfg.get("base_model_name_or_path")
    if not base:
        return model_id

    mtime = str(int(adapter_weights.stat().st_mtime))
    key = hashlib.sha1((str(p.resolve()) + "|" + mtime).encode("utf-8")).hexdigest()[:12]

    out_root = Path(
        os.environ.get(
            "AUTOMINGO_MERGED_DIR",
            str(Path.home() / ".cache" / "automingo" / "merged_models"),
        )
    )
    out_dir = out_root / f"{p.name}_merged_{key}"

    # Already merged
    if (out_dir / "config.json").exists():
        return str(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if hf_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif hf_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Qwen3-VL is a vision-to-seq model; this is the correct class for merging your adapter.
    model = AutoModelForVision2Seq.from_pretrained(
        base,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(p))
    model = model.merge_and_unload()
    model.save_pretrained(str(out_dir), safe_serialization=True)

    # Save tokenizer / processor next to merged weights
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    processor.save_pretrained(str(out_dir))

    # If adapter dir has a custom chat template, carry it over
    tpl = p / "chat_template.jinja"
    if tpl.exists():
        (out_dir / "chat_template.jinja").write_text(
            tpl.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    return str(out_dir)