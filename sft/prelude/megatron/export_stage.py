"""Export a Prelude mid-training stage from Levanter's packed loader to plain arrays for Megatron.

Reads the runtime's own stage_dataset() (ChatDataset packing, max 64 documents per sequence, seed-20261017
bijective shuffle), so row i is Levanter's i-th training sequence of the stage. No re-tokenization.

Outputs in OUT/:
  tokens.npy      int32 [N, L]   token ids; padding after the last document is pad id 0 as in Levanter
  loss_mask.npy   uint8 [N, L]   Levanter loss_weight: position t weights the prediction of token t+1
  cu_seqlens.npy  int32 [N, 65]  document starts plus the end of the last document, padded with -1
  manifest.json   written by `verify`

  prepare CACHE TOKENIZER CONTEXT OUT              create full-size arrays
  fill    CACHE TOKENIZER CONTEXT OUT START STOP   write rows [START, STOP)
  verify  CACHE TOKENIZER CONTEXT OUT              check totals against the cache, hash, write manifest
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

MAX_DOCS = 64
BATCH = 64


def open_arrays(out, mode, n=None, length=None):
    out = Path(out)
    if mode == "w+":
        out.mkdir(parents=True, exist_ok=False)
        return (np.lib.format.open_memmap(out / "tokens.npy", "w+", np.int32, (n, length)),
                np.lib.format.open_memmap(out / "loss_mask.npy", "w+", np.uint8, (n, length)),
                np.lib.format.open_memmap(out / "cu_seqlens.npy", "w+", np.int32, (n, MAX_DOCS + 1)))
    return tuple(np.load(out / f, mmap_mode=mode) for f in ("tokens.npy", "loss_mask.npy", "cu_seqlens.npy"))


def dataset(cache, tokenizer, context):
    from prepare_midtrain_levanter import stage_dataset
    return stage_dataset(cache, tokenizer, context)


def arr(x):
    return np.asarray(x.array if hasattr(x, "array") else x)


def boundaries(seg):
    real = np.where(seg >= 0)[0]
    end = int(real[-1]) + 1
    if not (seg[:end] >= 0).all() or (seg[end:] >= 0).any():
        raise ValueError("padding is not confined to the tail")
    starts = [0] + [int(i) for i in np.where(seg[1:end] != seg[:end - 1])[0] + 1]
    if len(starts) > MAX_DOCS:
        raise ValueError(f"{len(starts)} documents > {MAX_DOCS}")
    return starts + [end]


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    mode, cache, tokenizer, context, out = sys.argv[1:6]
    context = int(context)
    from levanter.utils.thread_utils import blocking_wait
    ds = dataset(cache, tokenizer, context)
    n = blocking_wait(ds.async_len())
    if mode == "prepare":
        open_arrays(out, "w+", n, context)
        print(f"prepared {out}: {n} x {context}")
        return
    if mode == "fill":
        start, stop = int(sys.argv[6]), min(int(sys.argv[7]), n)
        tokens, mask, cu = open_arrays(out, "r+")
        t0 = time.time()
        for b in range(start, stop, BATCH):
            idx = list(range(b, min(b + BATCH, stop)))
            for i, ex in zip(idx, blocking_wait(ds.get_batch(idx))):
                seg = ex.attn_mask.segment_ids
                seg = arr(seg[0] if isinstance(seg, tuple) else seg)
                w = arr(ex.loss_weight)
                if not np.isin(w, (0.0, 1.0)).all():
                    raise ValueError(f"row {i}: non-binary loss weight")
                bounds = boundaries(seg)
                tokens[i] = arr(ex.tokens)
                mask[i] = w.astype(np.uint8)
                cu[i] = -1
                cu[i, :len(bounds)] = bounds
            print(f"rows {b}..{idx[-1]} done, {(idx[-1] + 1 - start) / (time.time() - t0):.1f} rows/s", flush=True)
        for a in (tokens, mask, cu):
            a.flush()
        return
    if mode == "verify":
        tokens, mask, cu = open_arrays(out, "r")
        if tokens.shape != (n, context):
            raise ValueError(f"shape {tokens.shape} != {(n, context)}")
        ends = cu.max(axis=1)
        if (ends <= 0).any():
            raise ValueError(f"{int((ends <= 0).sum())} rows were never filled")
        stats = json.loads((Path(cache) / "train/.stats.json").read_text())
        counts = json.loads((Path(cache).parent / Path(cache).name / "import_counts.json").read_text()) \
            if (Path(cache) / "import_counts.json").exists() else {}
        seen = int(ends.sum(dtype=np.int64))
        trained = int(mask.sum(dtype=np.int64))
        docs = int((cu >= 0).sum() - n)
        result = {
            "rows": n, "context": context, "documents": docs, "seen_tokens": seen, "trained_positions": trained,
            "cache_rows": stats["total_elements"], "cache_tokens": stats["total_tokens"],
            "cache_trainable_tokens": counts.get("trainable_tokens"),
        }
        problems = []
        if docs != stats["total_elements"]:
            problems.append(f"documents {docs} != cache rows {stats['total_elements']}")
        if seen != stats["total_tokens"]:
            problems.append(f"seen tokens {seen} != cache tokens {stats['total_tokens']}")
        result["problems"] = problems
        manifest = {
            "description": "Levanter stage_dataset() packed rows for Megatron; row i = Levanter's i-th sequence",
            "source_cache": str(Path(cache).resolve()), "tokenizer": tokenizer, "shuffle_seed": 20261017,
            "conventions": {
                "tokens": "int32 [N, L]; tail padding uses pad id 0",
                "loss_mask": "uint8 [N, L]; position t weights the prediction of token t+1 (Levanter loss_weight)",
                "cu_seqlens": f"int32 [N, {MAX_DOCS + 1}]; document starts then end of the last document, -1 padded",
            },
            "counts": result,
            "sha256": {f: sha256(Path(out) / f) for f in ("tokens.npy", "loss_mask.npy", "cu_seqlens.npy")},
            "exporter_sha256": sha256(__file__),
        }
        (Path(out) / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(json.dumps(result, indent=1))
        if problems:
            sys.exit(1)


if __name__ == "__main__":
    main()
