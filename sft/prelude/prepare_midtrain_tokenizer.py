#!/usr/bin/env python3
"""Pin Prelude tokenizer/model compatibility; initialize six existing padding rows.

No model import or GPU is needed. Safetensors are patched in new files: only the
six embedding/head rows change. Unchanged shards are symlinks to the frozen base.
``export-tokenizer`` copies the frozen tokenizer into a completed HF export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/data"))
from prelude_midtrain import TOKEN_IDS, dump

TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "chat_template.jinja", "chat_tokenizer_manifest.json", "generation_config.json")
NEW_IDS = list(range(262145, 262151))
MATRICES = ("model.embed_tokens.weight", "lm_head.weight")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(16 * 1024**2):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def check_tokenizer(folder, *, source=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    for token, expected in TOKEN_IDS.items():
        if tok.convert_tokens_to_ids(token) != expected or tok(token, add_special_tokens=False)["input_ids"] != [expected]:
            raise ValueError(f"token ID mismatch: {token}")
    if len(tok) != 262151 or tok.bos_token_id != 1 or tok.eos_token_id not in ((2, 262146) if source else (262146,)) or tok.pad_token_id != 262144:
        raise ValueError("unexpected tokenizer vocabulary/configuration")
    return tok


def freeze(step1, out, model):
    step1, out, model = Path(step1), Path(out), Path(model)
    selection = json.loads((step1 / "source_selection.json").read_text())
    if selection["status"] != "step1-complete" or selection["license_policy"] != "all":
        raise ValueError("expected completed, permissive + no_license Step 1")
    for rel, expected in selection["artifact_sha256"].items():
        if sha(step1 / rel) != expected:
            raise ValueError(f"Step 1 artifact changed: {rel}")
    core = json.loads((step1 / "census_v2/verified_input_manifest.json").read_text())
    tokpath = Path(core["tokenizer"])
    check_tokenizer(tokpath, source=True)
    source_hashes = {name: sha(tokpath / name) for name in TOKENIZER_FILES}
    if source_hashes != core["tokenizer_files_sha256"]:
        raise ValueError("tokenizer differs from Step 1")
    if sha(Path(__file__).with_name("chat_template.jinja")) != source_hashes["chat_template.jinja"]:
        raise ValueError("repository chat template differs from frozen tokenizer")
    cfg = json.loads((model / "config.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())
    matrices = {}
    for name in MATRICES:
        file = index["weight_map"][name]
        offset, h = header(model / file)
        meta = h[name]
        if meta["shape"] != [cfg["vocab_size"], cfg["hidden_size"]] or cfg["vocab_size"] <= max(NEW_IDS):
            raise ValueError(f"incompatible matrix: {name}")
        if meta["dtype"] not in ("BF16", "F32", "F16"):
            raise ValueError("unsupported initialization dtype")
        matrices[name] = {"file": file, **meta, "header_sha256": hashlib.sha256(dump(h).encode()).hexdigest()}
    # Transformers 5 save_pretrained left the copied special_tokens_map stale.
    # Correct metadata only; tokenizer.json/model and the template stay byte-identical.
    special = json.loads((tokpath / "special_tokens_map.json").read_text())
    if isinstance(special["eos_token"], dict): special["eos_token"]["content"] = "<|im_end|>"
    else: special["eos_token"] = "<|im_end|>"
    special_bytes = (json.dumps(special, indent=2, sort_keys=True) + "\n").encode()
    tokenizer_hashes = {**source_hashes, "special_tokens_map.json": hashlib.sha256(special_bytes).hexdigest()}
    contract = {"schema_version": 1, "step1": str(step1.resolve()), "step1_sha256": sha(step1 / "source_selection.json"),
                "base_model": str(model.resolve()), "base_revision": "9d252ade085f74176eb2a3fa321a59f3b5af6deb",
                "base_config_sha256": sha(model / "config.json"),
                "base_index_sha256": sha(model / "model.safetensors.index.json"),
                "model_vocab_size": cfg["vocab_size"], "tokenizer_length": 262151,
                "token_ids": TOKEN_IDS, "tokenizer_files_sha256": tokenizer_hashes, "matrices": matrices,
                "source_tokenizer_files_sha256": source_hashes,
                "metadata_fix": "special_tokens_map.json EOS -> <|im_end|> to match tokenizer_config.json; no tokenization/template change",
                "initialization": {"ids": NEW_IDS, "method": "per-matrix mean of native rows [0,262144), float64 accumulation; round to storage dtype",
                                   "native_rows": [0, 262144], "pad_row_unchanged": True, "resize": False,
                                   "apply_once": "fresh base HF import only; resume/export never reinitializes"},
                "mask_policy": "prelude_midtrain_v1; whole tool blocks incl headers/end/LF masked; ordinary targets trained; BOS untrained",
                "document_eos_id": 2, "conversation_eos_id": 262146,
                "packing": {"mask_user_turns": True, "slice_strategy": "raise", "block_cross_document_attention": True,
                            "padding": "ChatDataset zero-pads IDs and masks; never train padding or next-document BOS"}}
    path = out / "tokenizer_contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("contract exists with different inputs")
    if path.exists():
        for name, expected in tokenizer_hashes.items():
            if sha(out / "tokenizer" / name) != expected:
                raise ValueError("existing frozen bundle changed")
        return contract
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "tokenizer"
    dest.mkdir(exist_ok=True)
    for name in TOKENIZER_FILES:
        if name == "special_tokens_map.json": (dest / name).write_bytes(special_bytes)
        else: shutil.copyfile(tokpath / name, dest / name)
    check_tokenizer(dest)
    write(path, contract)
    return contract


def encode_mean(mean, dtype):
    import numpy as np
    if dtype == "BF16":
        bits = np.asarray(mean, dtype="<f4").view("<u4")
        # IEEE round-to-nearest-even, as used for bfloat16 casts.
        return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype("<u2").tobytes()
    return mean.astype("<f4" if dtype == "F32" else "<f2").tobytes()


def matrix_mean(path, meta, native_end=262144):
    import numpy as np
    offset, h = header(path)
    if h[meta["name"]] != {k: meta[k] for k in ("shape", "dtype", "data_offsets")}:
        raise ValueError("matrix metadata changed")
    width = meta["shape"][1]
    dtype = "<u2" if meta["dtype"] == "BF16" else "<f4" if meta["dtype"] == "F32" else "<f2"
    total = np.zeros(width, dtype=np.float64)
    with open(path, "rb") as f:
        f.seek(offset + meta["data_offsets"][0])
        for start in range(0, native_end, 1024):
            size = min(1024, native_end - start)
            arr = np.frombuffer(f.read(size * width * np.dtype(dtype).itemsize), dtype=dtype).reshape(size, width)
            if meta["dtype"] == "BF16":
                arr = (arr.astype("<u4") << 16).view("<f4")
            if not np.isfinite(arr).all():
                raise ValueError("nonfinite native embedding")
            total += arr.sum(axis=0, dtype=np.float64)
    return encode_mean(total / native_end, meta["dtype"])


def initialize(contract_file, out):
    contract_file, out = Path(contract_file), Path(out)
    c = json.loads(contract_file.read_text())
    source = Path(c["base_model"])
    if out.exists():
        raise ValueError("initialization output must be new (prevents reinitializing a trained checkpoint)")
    for name, key in [("config.json", "base_config_sha256"), ("model.safetensors.index.json", "base_index_sha256")]:
        if sha(source / name) != c[key]:
            raise ValueError("base identity changed")
    # Validate every header before creating any output.
    for name, m in c["matrices"].items():
        if hashlib.sha256(dump(header(source / m["file"])[1]).encode()).hexdigest() != m["header_sha256"]:
            raise ValueError("base matrix header changed")
    out.mkdir(parents=True)
    changed = {m["file"] for m in c["matrices"].values()}
    for p in source.iterdir():
        if not p.is_file() or p.name in TOKENIZER_FILES or p.name == "config.json":
            continue
        if p.name in changed:
            shutil.copyfile(p, out / p.name)
        else:
            (out / p.name).symlink_to(p.resolve())
    audit = {}
    for name, m in c["matrices"].items():
        raw = matrix_mean(source / m["file"], {**m, "name": name})
        offset, _ = header(out / m["file"])
        with open(out / m["file"], "r+b") as f:
            for index in NEW_IDS:
                f.seek(offset + m["data_offsets"][0] + index * len(raw))
                f.write(raw)
        audit[name] = {"initialized_ids": NEW_IDS, "row_sha256": hashlib.sha256(raw).hexdigest()}
    # Compare all bytes, excluding only the explicitly initialized row ranges.
    for file in sorted(changed):
        offset, _ = header(source / file)
        ranges = []
        for m in c["matrices"].values():
            if m["file"] == file:
                width = m["shape"][1] * (4 if m["dtype"] == "F32" else 2)
                ranges.extend((offset + m["data_offsets"][0] + i * width, width) for i in NEW_IDS)
        with open(source / file, "rb") as a, open(out / file, "rb") as b:
            cursor = 0
            for start, length in sorted(ranges) + [(source.joinpath(file).stat().st_size, 0)]:
                while cursor < start:
                    n = min(16 * 1024**2, start - cursor)
                    if a.read(n) != b.read(n):
                        raise ValueError("non-initialized model bytes changed")
                    cursor += n
                a.seek(length, 1); b.seek(length, 1); cursor += length
    cfg = json.loads((source / "config.json").read_text())
    cfg.update(bos_token_id=1, pad_token_id=262144, eos_token_id=[262146, 2])
    write(out / "config.json", cfg)
    export_tokenizer(contract_file, out)
    write(out / "midtrain_initialization.json", {"contract_sha256": sha(contract_file), "matrices": audit,
          "verified_all_other_model_bytes_unchanged": True, "unchanged_shards": "symlinks to frozen base",
          "shard_sha256": {f: sha(out / f) for f in changed}})


def export_tokenizer(contract_file, checkpoint):
    contract_file, checkpoint = Path(contract_file), Path(checkpoint)
    c = json.loads(contract_file.read_text())
    cfg = json.loads((checkpoint / "config.json").read_text())
    if cfg["vocab_size"] != c["model_vocab_size"]:
        raise ValueError("exported checkpoint vocabulary differs from frozen contract")
    for name, expected in c["tokenizer_files_sha256"].items():
        folder = contract_file.parent / "tokenizer"
        if not folder.is_dir(): folder = contract_file.parent  # contract in an exported HF checkpoint
        source = folder / name
        if sha(source) != expected:
            raise ValueError("frozen tokenizer changed")
        if source.resolve() != (checkpoint / name).resolve(): shutil.copyfile(source, checkpoint / name)
    if contract_file.resolve() != (checkpoint / "tokenizer_contract.json").resolve():
        shutil.copyfile(contract_file, checkpoint / "tokenizer_contract.json")
    cfg.update(bos_token_id=1, pad_token_id=262144, eos_token_id=[262146, 2])
    write(checkpoint / "config.json", cfg)


def install_checkpoint_hooks(contract_file):
    """Install only in the dedicated midtrain trainer process, before HF import.

    Levanter normally shrinks the padded model vocabulary to tokenizer length.
    Preserve all 262272 model rows while keeping the frozen tokenizer at 262151.
    Each local HF export gets the byte-identical frozen bundle and contract.
    """
    from functools import wraps
    from levanter.compat.hf_checkpoints import HFCheckpointConverter
    import jax
    contract_file = Path(contract_file)
    c = json.loads(contract_file.read_text())
    # train_lm constructs the resume exemplar from tokenizer length before it
    # loads native checkpoints. HF import alone can preserve padding on a fresh
    # run, but the resume exemplar must use the same padded axis too.
    import levanter.main.train_lm as train_lm
    vocab_identity = sha(contract_file)
    previous_vocab_identity = getattr(train_lm, '_prelude_vocab_contract_sha256', None)
    if previous_vocab_identity and previous_vocab_identity != vocab_identity:
        raise ValueError('different Prelude training vocabulary already installed')
    if not previous_vocab_identity:
        original_round = train_lm.round_axis_for_partitioning

        def round_training_axis(axis, mapping):
            if axis.name != 'vocab':
                return original_round(axis, mapping)
            if axis.size not in {c['tokenizer_length'], c['model_vocab_size']}:
                raise ValueError('unexpected Prelude training vocabulary')
            result = original_round(axis.resize(c['model_vocab_size']), mapping)
            if result.size != c['model_vocab_size']:
                raise ValueError('mesh must divide the frozen padded Prelude vocabulary')
            return result

        train_lm.round_axis_for_partitioning = round_training_axis
        train_lm._prelude_vocab_contract_sha256 = vocab_identity
    cls = HFCheckpointConverter
    identity = sha(contract_file)
    if getattr(cls, "_prelude_contract_sha256", None):
        if cls._prelude_contract_sha256 != identity: raise ValueError("different Prelude export contract already installed")
        return
    load, save = cls.load_pretrained, cls.save_pretrained

    @wraps(load)
    def load_preserving_vocab(self, *args, **kwargs):
        if self.default_hf_config.vocab_size != c["model_vocab_size"]:
            raise ValueError("HF model vocabulary differs from Prelude contract")
        kwargs["resize_vocab_to_match_tokenizer"] = False
        model = load(self, *args, **kwargs)
        if model.Vocab.size != c["model_vocab_size"]: raise ValueError("Prelude model vocabulary changed during HF import")
        return model

    @wraps(save)
    def save_with_frozen_tokenizer(self, model, path, *args, **kwargs):
        if "://" in str(path): raise ValueError("Prelude checkpoint hook requires a local HF export path")
        result = save(self, model, path, *args, **kwargs)
        if jax.process_index() == 0: export_tokenizer(contract_file, Path(path))
        return result

    cls.load_pretrained, cls.save_pretrained = load_preserving_vocab, save_with_frozen_tokenizer
    cls._prelude_contract_sha256 = identity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--step1", required=True); p.add_argument("--out", required=True); p.add_argument("--model", required=True)
    p = sub.add_parser("initialize-hf")
    p.add_argument("--contract", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("export-tokenizer")
    p.add_argument("--contract", required=True); p.add_argument("--checkpoint", required=True)
    a = ap.parse_args()
    if a.command == "freeze": freeze(a.step1, a.out, a.model)
    elif a.command == "initialize-hf": initialize(a.contract, a.out)
    else: export_tokenizer(a.contract, a.checkpoint)


if __name__ == "__main__":
    main()
