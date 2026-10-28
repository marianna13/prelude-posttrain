"""Export a Megatron-Bridge mid-training checkpoint (megatron_midtrain.py) to a HuggingFace directory.

Loads the iteration on CPU at TP1 (Bridge resets the parallel layout), writes safetensors + the base model's
tokenizer/config files, then sets the trained RoPE theta and context in config.json, because Bridge copies the
base config unchanged. Tokenizer files Bridge drops or rewrites are copied from the base unchanged.
Usage: python export_hf.py BASE_HF CKPT_ITER_DIR OUT_DIR
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import megatron_midtrain  # noqa: F401  version shims only; training runs under __main__
import yaml
from megatron.bridge import AutoBridge

base, ckpt, out = sys.argv[1:4]
run_cfg = yaml.safe_load(open(Path(ckpt) / "run_config.yaml"))["model"]
theta, context = float(run_cfg["rotary_base"]), int(run_cfg["max_position_embeddings"])
print("checkpoint rotary_base", theta, "max_position_embeddings", context, flush=True)

AutoBridge.from_hf_pretrained(base).export_ckpt(ckpt, out, show_progress=False)

cfg_path = Path(out) / "config.json"
cfg = json.load(open(cfg_path))
cfg["rope_theta"] = theta
cfg["rope_parameters"] = {**(cfg.get("rope_parameters") or {}), "rope_theta": theta}
if run_cfg.get("position_embedding_type") == "yarn":  # factors come from YARN_FACTOR / YARN_ORIG (not saved)
    assert megatron_midtrain.YARN_FACTOR, "YaRN checkpoint: set YARN_FACTOR and YARN_ORIG"
    cfg["rope_parameters"] = {"rope_type": "yarn", "rope_theta": theta, "factor": megatron_midtrain.YARN_FACTOR,
                              "original_max_position_embeddings": megatron_midtrain.YARN_ORIG}
cfg["max_position_embeddings"] = context
json.dump(cfg, open(cfg_path, "w"), indent=2)
for name in ["tokenizer_config.json", "tokenizer.model", "added_tokens.json", "special_tokens_map.json",
             "chat_tokenizer_manifest.json", "tokenizer_contract.json"]:
    if (Path(base) / name).exists():
        shutil.copyfile(Path(base) / name, Path(out) / name)
for f in Path(out).iterdir():
    os.chmod(f, 0o644)  # Bridge writes safetensors as 0600
json.dump({"megatron_checkpoint": str(ckpt), "base_hf": base, "rope_theta": theta,
           "max_position_embeddings": context}, open(Path(out) / "megatron_export.json", "w"), indent=2)
print("exported", out, flush=True)
