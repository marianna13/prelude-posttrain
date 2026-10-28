"""Build the Prelude chat tokenizer: Prelude's tokenizer + ChatML/think/tool tokens + chat_template.jinja.

usage: python make_chat_tokenizer.py PRELUDE_MODEL_DIR OUT_DIR

- Adds <|im_start|>, <|im_end|> (special) and <think>, </think>, <tool_response>, </tool_response> (not special,
  as in Qwen3). Prelude's vocab is padded to 262,272 rows (256k tokenizer = 262,145 ids), so the new ids
  (262,145+) fall into existing, untrained embedding rows: no resize, but the SFT should initialize them
  (e.g. to the mean embedding) or they start from whatever the padding rows hold.
- Un-flags the existing <tool_call>/</tool_call> (ids 11/12) as special: vLLM drops special tokens from
  outputs, which would strip the tags the hermes tool parser looks for.
- eos_token becomes <|im_end|> (end of an assistant turn); generation_config lists <|im_end|> and <eos>.
"""

import json
import os
import shutil
import sys

from transformers import AutoTokenizer

src, out = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out, exist_ok=True)
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "added_tokens.json"):
    if os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))

# 1. <tool_call>/</tool_call>: not special (tokenizer.json added_tokens + config + special map)
tj = json.load(open(os.path.join(out, "tokenizer.json")))
for t in tj["added_tokens"]:
    if t["content"] in ("<tool_call>", "</tool_call>"):
        t["special"] = False
json.dump(tj, open(os.path.join(out, "tokenizer.json"), "w"), ensure_ascii=False)
for name in ("tokenizer_config.json", "special_tokens_map.json"):
    p = os.path.join(out, name)
    cfg = json.load(open(p))
    for t in (cfg.get("added_tokens_decoder") or {}).values():
        if t["content"] in ("<tool_call>", "</tool_call>"):
            t["special"] = False
    for key in ("additional_special_tokens", "extra_special_tokens"):
        if isinstance(cfg.get(key), list):
            cfg[key] = [t for t in cfg[key] if (t if isinstance(t, str) else t.get("content")) not in ("<tool_call>", "</tool_call>")]
    json.dump(cfg, open(p, "w"), indent=2, ensure_ascii=False)

# 2. new tokens + template
tok = AutoTokenizer.from_pretrained(out)
n0 = len(tok)
tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}, replace_extra_special_tokens=False)
tok.add_tokens(["<think>", "</think>", "<tool_response>", "</tool_response>"], special_tokens=False)
tok.chat_template = open(os.path.join(here, "chat_template.jinja")).read()
tok.eos_token = "<|im_end|>"
tok.save_pretrained(out)
# Transformers 5 can leave an existing copied special_tokens_map.json untouched.
# Keep it consistent with tokenizer_config.json (including tool-call visibility).
special = json.load(open(os.path.join(out, "special_tokens_map.json")))
special["eos_token"] = "<|im_end|>"
json.dump(special, open(os.path.join(out, "special_tokens_map.json"), "w"), indent=2, ensure_ascii=False)

ids = {t: tok.convert_tokens_to_ids(t) for t in
       ("<bos>", "<eos>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>")}
assert max(ids.values()) < 262272, ids
json.dump({"bos_token_id": ids["<bos>"], "eos_token_id": [ids["<|im_end|>"], ids["<eos>"]], "pad_token_id": tok.pad_token_id},
          open(os.path.join(out, "generation_config.json"), "w"), indent=2)
json.dump({"source_tokenizer": os.path.abspath(src), "added": {k: v for k, v in ids.items() if v >= n0},
           "token_ids": ids, "vocab_before": n0, "vocab_after": len(tok)},
          open(os.path.join(out, "chat_tokenizer_manifest.json"), "w"), indent=2)
print(json.dumps(ids))
