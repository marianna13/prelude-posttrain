#!/usr/bin/env python3
"""Pin a bounded local replay cut and additional long PDFs, then census that cut.

This is an explicitly documented approximation of the unavailable native blend.
It never counts a bounded cut as a census of the full upstream dataset.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys

DEPS = "/p/project1/laionize/marianna/dc_agent/.prelude_census_deps"
TOKENIZER = "/p/data1/mmlaion/marianna/sft_sources/prelude_chat_tokenizer/tokenizer.json"
LANGUAGES = ["bul_Cyrl", "ces_Latn", "dan_Latn", "deu_Latn", "ell_Grek", "ekk_Latn", "fin_Latn",
             "fra_Latn", "gle_Latn", "hrv_Latn", "hun_Latn", "ita_Latn", "lvs_Latn", "lit_Latn",
             "mlt_Latn", "nld_Latn", "pol_Latn", "por_Latn", "ron_Latn", "slk_Latn", "slv_Latn",
             "spa_Latn", "swe_Latn"]
_TOKENIZER = None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(16 * 1024**2):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)


def state(path):
    st = Path(path).stat()
    return {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def metadata_pin(path, root):
    for directory in (".cache/huggingface/download", ".huggingface/download"):
        metadata = root / directory / (path.relative_to(root).as_posix() + ".metadata")
        if metadata.exists():
            lines = metadata.read_text().splitlines()
            return {"revision": lines[0], "upstream_etag": lines[1]}
    return {"revision": None, "upstream_etag": None}


def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    def add(path, root, source, hf_repo, family, language, limit):
        pin = metadata_pin(path, root)
        entries.append(dict(file_id=len(entries), path=str(path), relative_path=path.relative_to(root).as_posix(),
                            source=source, hf_repo=hf_repo, family=family, language=language,
                            row_limit=limit, text_field="text", state=state(path), **pin))
    dclm = Path("/p/data1/mmlaion/mlfoundations-dclm-baseline-1.0")
    paths = list(dclm.glob("global-shard*/local-shard*/*.zst"))
    paths = sorted(paths, key=lambda p: hashlib.sha256(p.relative_to(dclm).as_posix().encode()).hexdigest())[:16]
    if len(paths) != 16:
        raise ValueError("Need 16 local DCLM files")
    for p in paths:
        add(p, dclm, "dclm-baseline-1.0", "mlfoundations/dclm-baseline-1.0", "dclm_english", "eng_Latn", 100000)
    nemo = Path("/p/data1/datasets/Nemotron-CC/nemotron_cc_data/data")
    index = Path("/p/data1/datasets/Nemotron-CC/nemotron_cc_data/files_to_download.paths")
    paths = [nemo / line.strip() for line in index.read_text().splitlines()
             if "/quality=high/kind=actual/kind2=actual/" in line and line.endswith(".zstd")]
    paths = sorted(paths, key=lambda p: hashlib.sha256(p.relative_to(nemo).as_posix().encode()).hexdigest())[:8]
    if len(paths) != 8 or not all(p.is_file() for p in paths):
        raise ValueError("Need 8 local Nemotron-CC high-quality original files")
    for p in paths:
        add(p, nemo, "Nemotron-CC-original-high", "nvidia/Nemotron-CC", "nemotron_english", "eng_Latn", 100000)
    pdf = Path("/p/data1/datasets/finepdfs")
    english = sorted((pdf / "data/eng_Latn/train").glob("*.parquet"))
    if len(english) < 2:
        raise ValueError("Need disjoint English PDF files for replay and long documents")
    add(english[0], pdf, "FinePDFs-English-long", "HuggingFaceFW/finepdfs", "long_documents", "eng_Latn", 0)
    add(english[1], pdf, "FinePDFs-English-replay", "HuggingFaceFW/finepdfs", "pdf_english", "eng_Latn", 0)
    for lang in LANGUAGES:
        paths = sorted((pdf / "data" / lang / "train").glob("*.parquet"))
        if not paths:
            raise ValueError(f"Missing local FinePDFs language: {lang}")
        add(paths[0], pdf, "FinePDFs-EU-replay", "HuggingFaceFW/finepdfs", "pdf_multilingual", lang, 50000)
    # Retain the native blend's reported 18% non-English share. Rescale the three
    # available English prose families from 33.2:24.5:10.9 to the remaining 82%.
    denom = 33.2 + 24.5 + 10.9
    weights = {"dclm_english": .82*33.2/denom, "nemotron_english": .82*24.5/denom,
               "pdf_english": .82*10.9/denom, "pdf_multilingual": .18}
    manifest = dict(schema_version=1, entries=entries, family_weights=weights,
                    tokenizer=TOKENIZER, tokenizer_sha256=sha256(TOKENIZER),
                    implementation_sha256=sha256(__file__),
                    source_scope="Exact first N records of deterministic SHA256-ranked file selections; 0 means full file",
                    replay_eligibility="nonempty text, rendered length <=4096 tokens including BOS/EOS",
                    long_document_eligibility="nonempty text, rendered length >=32768 tokens including BOS/EOS",
                    language_weights="Within non-English family, proportional to available eligible tokens",
                    differences_from_native=[
                        "Native Leonardo/LUMI mixture file is unavailable here",
                        "Nemotron-CC uses only high-quality original documents in this proxy",
                        "Non-English HPLT replay is approximated by FinePDFs in 23 EU languages",
                        "StarCoder, OLMo, and the native math/translation components are not reproduced",
                        "The local StarCoder aggregate lacks sufficient file-level provenance for this cut",
                    ],
                    provenance_note="Null HF revision is pinned by the locally computed file SHA256, not asserted to match current HF main",
                    dependencies={name: importlib.metadata.version(name) for name in
                                  ("tokenizers", "pyarrow", "zstandard")})
    manifest["build_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    p = out / "input_manifest.json"
    if p.exists() and json.loads(p.read_text())["build_id"] != manifest["build_id"]:
        raise ValueError("Different replay manifest already exists")
    write_json(p, manifest)
    print(json.dumps({"files":len(entries),"bytes":sum(e['state']['bytes'] for e in entries),
                      "out":str(out),"family_weights":weights}), flush=True)


def init_worker():
    global _TOKENIZER
    sys.path.insert(0, DEPS)
    import pyarrow as pa
    from tokenizers import Tokenizer
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    _TOKENIZER = Tokenizer.from_file(TOKENIZER)


def records(entry):
    path = entry["path"]
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=[entry["text_field"]], use_threads=False):
            for row in batch.to_pylist():
                yield row.get(entry["text_field"]), None
    else:
        import zstandard
        with open(path,"rb") as f:
            with zstandard.ZstdDecompressor().stream_reader(f,read_across_frames=True) as reader:
                for line in io.BufferedReader(reader):
                    try:
                        row=json.loads(line)
                        yield row.get(entry["text_field"]), None
                    except (ValueError,UnicodeError,AttributeError):
                        yield None,"invalid_json"


def process_file(job):
    import pyarrow as pa
    import pyarrow.parquet as pq
    entry,out,build_id = job
    out=Path(out)
    key=f"{entry['file_id']:04d}"
    result_path=out/(key+'.json')
    if result_path.exists() and json.loads(result_path.read_text()).get("build_id")==build_id:
        return json.loads(result_path.read_text())
    if state(entry["path"])!=entry["state"]:
        raise ValueError("Input changed: "+entry["path"])
    digest=sha256(entry["path"])
    expected=entry.get('upstream_etag') or ''
    if len(expected)==64 and expected!=digest:
        raise ValueError("Upstream checksum mismatch: "+entry["path"])
    target=out/(key+'.parquet')
    writer=None
    buffered, texts=[],[]
    counts=dict(rows=0,rejected=0,valid=0,tokens=0,eligible_rows=0,eligible_tokens=0)
    def flush():
        nonlocal writer,texts,buffered
        if not texts:return
        encoded=_TOKENIZER.encode_batch([x[1] for x in texts],add_special_tokens=False)
        for (index,text),encoding in zip(texts,encoded):
            tokens=len(encoding.ids)+2
            eligible=tokens>=32768 if entry['family']=='long_documents' else tokens<=4096
            buffered.append(dict(file_id=entry['file_id'],row=index,source=entry['source'],
                                 family=entry['family'],language=entry['language'],tokens=tokens,eligible=eligible))
            counts['valid']+=1;counts['tokens']+=tokens
            if eligible:counts['eligible_rows']+=1;counts['eligible_tokens']+=tokens
        texts=[]
        if len(buffered)>=4096:
            table=pa.Table.from_pylist(buffered)
            if writer is None:writer=pq.ParquetWriter(str(target)+'.tmp',table.schema,compression='zstd')
            writer.write_table(table);buffered=[]
    try:
        for index,(text,error) in enumerate(records(entry)):
            if entry['row_limit'] and index>=entry['row_limit']:break
            counts['rows']+=1
            if error or not isinstance(text,str) or not text.strip():
                counts['rejected']+=1;continue
            texts.append((index,text))
            if len(texts)>=128 or sum(len(t) for _,t in texts)>=2*1024**2:flush()
        flush()
        if buffered:
            table=pa.Table.from_pylist(buffered)
            if writer is None:writer=pq.ParquetWriter(str(target)+'.tmp',table.schema,compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None:writer.close()
    if writer is not None:os.replace(str(target)+'.tmp',target)
    if state(entry["path"])!=entry["state"]:raise ValueError("Input changed during census")
    result=dict(file_id=entry['file_id'],build_id=build_id,sha256=digest,counts=counts,
                family=entry['family'],language=entry['language'],path=entry['path'],
                revision=entry['revision'],scope_rows=entry['row_limit'])
    write_json(result_path,result)
    return result


def run(out,workers):
    out=Path(out)
    manifest=json.loads((out/'input_manifest.json').read_text())
    if sha256(__file__)!=manifest['implementation_sha256'] or sha256(TOKENIZER)!=manifest['tokenizer_sha256']:
        raise ValueError('Implementation/tokenizer changed since prepare')
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=init_worker) as pool:
        futures=[pool.submit(process_file,(entry,str(out),manifest['build_id'])) for entry in manifest['entries']]
        results=[]
        for future in as_completed(futures):
            r=future.result();results.append(r)
            print(json.dumps(dict(file_id=r['file_id'],family=r['family'],language=r['language'],counts=r['counts'])),flush=True)
    eligible={family:sum(r['counts']['eligible_tokens'] for r in results if r['family']==family)
              for family in manifest['family_weights']}
    if any(n==0 for n in eligible.values()):raise ValueError('A replay family has no eligible tokens')
    capacity=min(eligible[f]/w for f,w in manifest['family_weights'].items())
    language_weights={r['language']:r['counts']['eligible_tokens']/eligible['pdf_multilingual']
                      for r in results if r['family']=='pdf_multilingual'}
    verified=dict(manifest,entries=[dict(e,sha256=next(r['sha256'] for r in results if r['file_id']==e['file_id']))
                                   for e in manifest['entries']],status='complete')
    write_json(out/'verified_input_manifest.json',verified)
    write_json(out/'summary.json',dict(status='complete',build_id=manifest['build_id'],files=len(results),
                                     family_weights=manifest['family_weights'],eligible_tokens=eligible,
                                     max_replay_tokens_without_family_repetition=capacity,
                                     multilingual_language_weights=language_weights,results=sorted(results,key=lambda r:r['file_id'])))


def main():
    sys.path.insert(0,DEPS)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--out',required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    prepare(args.out) if args.command=='prepare' else run(args.out,args.workers)


if __name__=='__main__':main()
