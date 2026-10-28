#!/usr/bin/env python3
"""Select long Stack repositories using exact UltraData-L2 content-ID evidence.

Reads only census metadata, never source text. All thresholds and sort keys are
recorded, and selections refer back to frozen file/row/commit identifiers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

DEFAULT_DEPS = "/p/project1/laionize/marianna/dc_agent/.prelude_census_deps"


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temp, path)


def select_repos(out, license_policy="permissive", min_tokens=8192, min_density=0.2,
                 threads=32, memory="128GB", require_complete=True):
    import duckdb

    out = Path(out)
    manifest = json.loads((out / "input_manifest.json").read_text())
    if require_complete:
        report = json.loads((out / "census_summary.json").read_text())
        if report["status"] != "complete" or report["build_id"] != manifest["build_id"]:
            raise ValueError("A complete matching census is required")
        for entry in manifest["files"]:
            current = Path(entry["path"]).stat()
            if {"bytes":current.st_size,"mtime_ns":current.st_mtime_ns} != entry["state"]:
                raise ValueError("Frozen input changed after census: " + entry["path"])
        for name, expected in manifest["tokenizer_files_sha256"].items():
            if hashlib.sha256((Path(manifest["tokenizer"])/name).read_bytes()).hexdigest()!=expected:
                raise ValueError("Tokenizer changed after census: " + name)
    connection = duckdb.connect()
    connection.execute(f"SET threads={int(threads)}")
    connection.execute("SET memory_limit='" + memory.replace("'", "''") + "'")
    temp = out / "selection_duckdb_tmp"
    temp.mkdir(exist_ok=True)
    connection.execute("SET temp_directory='" + str(temp).replace("'", "''") + "'")
    connection.read_parquet(str(out / "metrics/*.parquet")).create_view("metrics")
    connection.read_parquet(str(out / "ultra_files/*.parquet")).create_view("ultra")
    connection.read_parquet(str(out / "stack_files/*.parquet")).create_view("stack")
    for table in ("ultra", "stack"):
        unexpected = connection.execute(f"""SELECT license,count(*) FROM {table}
            WHERE NOT is_vendor AND license NOT IN ('permissive','no_license') GROUP BY license""").fetchall()
        if unexpected:
            raise ValueError(f"Nonvendor token counts include unsupported licenses in {table}: {unexpected}")
    license_condition = "license='permissive'" if license_policy == "permissive" else "license IN ('permissive','no_license')"
    token_field = "permissive_tokens" if license_policy == "permissive" else "all_nonvendor_tokens"
    connection.execute(f"""CREATE TEMP TABLE l2 AS
        SELECT repo_key,content_id,max(quality_score) quality_score,max(algo_rel_score) algo_rel_score
        FROM ultra WHERE {license_condition} AND NOT is_vendor AND content_id<>''
        GROUP BY repo_key,content_id""")
    # Deduplicate the evidence before the join, so duplicate L2 rows cannot inflate file density.
    connection.execute(f"""CREATE TEMP TABLE evidence AS
        SELECT s.id,count(*) eligible_file_count,count(l.content_id) l2_matched_files,
               avg(l.quality_score) mean_l2_quality,avg(l.algo_rel_score) mean_l2_algorithmic_relevance,
               count(l.content_id)::DOUBLE/count(*) l2_file_density
        FROM (SELECT * FROM stack WHERE {license_condition} AND NOT is_vendor) s
        LEFT JOIN l2 l ON s.repo_key=l.repo_key AND s.content_id=l.content_id
        GROUP BY s.id""")
    connection.execute(f"""CREATE TEMP TABLE selected AS
        SELECT m.id,m.file_id,m."row",m.repo_key,m.repo_path,m.commit_id,m.stars,m.num_files,
               m.permissive_files,e.eligible_file_count,m.tokens raw_tokens,
               m.{token_field} selected_tokens,e.l2_matched_files,e.l2_file_density,
               e.mean_l2_quality,e.mean_l2_algorithmic_relevance,
               e.l2_file_density*e.mean_l2_quality selection_score,
               CASE WHEN m.{token_field}>=32768 THEN 'long_documents' ELSE 'code' END mix_category
        FROM metrics m JOIN evidence e USING(id)
        WHERE m.source='stack-v3-train' AND m.{token_field}>={int(min_tokens)}
              AND e.l2_matched_files>=1 AND e.l2_file_density>={float(min_density)}""")
    destination = out / "repo_selection"
    destination.mkdir(exist_ok=True)
    selected_path = destination / (license_policy + ".parquet")
    connection.sql("SELECT * FROM selected ORDER BY selection_score DESC,stars DESC,repo_key,commit_id,id").write_parquet(
        str(selected_path), compression="zstd")
    evidence_path = destination / (license_policy + "_evidence.parquet")
    connection.sql("SELECT * FROM evidence ORDER BY id").write_parquet(str(evidence_path), compression="zstd")
    stats = connection.execute("""SELECT count(*) n_repos,coalesce(sum(selected_tokens),0) tokens,
        coalesce(sum(selected_tokens-1),0) trainable_tokens,
        coalesce(max(selected_tokens),0) max_selected_tokens,
        coalesce(sum(eligible_file_count),0) files,coalesce(sum(l2_matched_files),0) l2_matched_files,
        quantile_cont(selected_tokens,[0.5,0.9,0.95,0.99]) token_percentiles,
        quantile_cont(l2_file_density,[0.5,0.9,0.99]) density_percentiles,
        quantile_cont(mean_l2_quality,[0.5,0.9,0.99]) quality_percentiles,
        quantile_cont(mean_l2_algorithmic_relevance,[0.5,0.9,0.99]) algorithmic_relevance_percentiles,
        quantile_cont(selection_score,[0.5,0.9,0.99]) score_percentiles,
        quantile_cont(stars,[0.5,0.9,0.99]) stars_percentiles FROM selected""").fetchdf().to_dict("records")[0]
    for key, value in stats.items():
        if hasattr(value, "tolist"):
            stats[key] = value.tolist()
    for key in ("n_repos", "tokens", "trainable_tokens", "max_selected_tokens", "files", "l2_matched_files"):
        stats[key] = int(stats[key])
    by_category = connection.execute("""SELECT mix_category,count(*) n_repos,sum(selected_tokens) tokens,
        sum(selected_tokens-1) trainable_tokens
        FROM selected GROUP BY mix_category ORDER BY mix_category""").fetchdf().to_dict("records")
    languages = connection.execute(f"""SELECT s.language,count(*) n_files FROM stack s JOIN selected r USING(id)
        WHERE s.{license_condition} AND NOT s.is_vendor GROUP BY s.language ORDER BY n_files DESC""").fetchdf().to_dict("records")
    all_license_groups = connection.execute("""SELECT license,is_vendor,count(*) n_files
        FROM stack GROUP BY license,is_vendor ORDER BY n_files DESC""").fetchdf().to_dict("records")
    policy_summary = connection.execute(f"""SELECT source,subset,count(*) raw_rows,sum(tokens) raw_tokens,
        count(*) FILTER(WHERE {token_field}>0) eligible_rows,sum({token_field}) eligible_tokens,
        sum(greatest({token_field}-1,0)) eligible_trainable_tokens,
        quantile_cont({token_field},[0.5,0.9,0.95,0.99]) FILTER(WHERE {token_field}>0) length_percentiles
        FROM metrics WHERE source IN ('stack-v3-train','UltraData-Stack-v3-L2')
        GROUP BY source,subset ORDER BY source,subset""").fetchdf().to_dict("records")
    for row in policy_summary:
        if hasattr(row["length_percentiles"], "tolist"):
            row["length_percentiles"] = row["length_percentiles"].tolist()
    def length_bin(field):
        return f"""CASE WHEN {field}<4096 THEN '<4K' WHEN {field}<8192 THEN '4-8K'
            WHEN {field}<16384 THEN '8-16K' WHEN {field}<32768 THEN '16-32K'
            WHEN {field}<=65536 THEN '32-64K' ELSE '>64K' END"""
    policy_bins = connection.execute(f"""SELECT source,subset,{length_bin(token_field)} length_bin,
        count(*) n_rows,sum({token_field}) tokens FROM metrics
        WHERE source IN ('stack-v3-train','UltraData-Stack-v3-L2') AND {token_field}>0
        GROUP BY source,subset,length_bin ORDER BY source,subset,length_bin""").fetchdf().to_dict("records")
    selected_bins = connection.execute(f"""SELECT mix_category,{length_bin('selected_tokens')} length_bin,
        count(*) n_repos,sum(selected_tokens) tokens FROM selected
        GROUP BY mix_category,length_bin ORDER BY mix_category,length_bin""").fetchdf().to_dict("records")
    gates = [
        ("license_vendor", f"m.{token_field}>0"),
        ("minimum_length", f"m.{token_field}>={int(min_tokens)}"),
        ("l2_match", f"m.{token_field}>={int(min_tokens)} AND e.l2_matched_files>=1"),
        ("l2_density", f"m.{token_field}>={int(min_tokens)} AND e.l2_matched_files>=1 AND e.l2_file_density>={float(min_density)}"),
    ]
    funnel, funnel_bins = [], []
    for stage, condition in gates:
        base = f"FROM metrics m LEFT JOIN evidence e USING(id) WHERE m.source='stack-v3-train' AND {condition}"
        row = connection.execute(f"""SELECT count(*) n_repos,coalesce(sum(m.{token_field}),0) tokens,
            coalesce(sum(m.{token_field}-1),0) trainable_tokens {base}""").fetchone()
        funnel.append(dict(stage=stage,n_repos=int(row[0]),tokens=int(row[1]),trainable_tokens=int(row[2])))
        rows = connection.execute(f"""SELECT {length_bin('m.'+token_field)} length_bin,
            count(*) n_repos,sum(m.{token_field}) tokens {base} GROUP BY length_bin""").fetchdf().to_dict("records")
        funnel_bins.extend(dict(row,stage=stage) for row in rows)
    result = {"build_id": manifest["build_id"], "license_policy": license_policy,
              "min_tokens": min_tokens, "min_l2_file_density": min_density, "min_l2_matched_files": 1,
              "exclude_vendor": True, "join_key": ["lowercase GitHub repository identity", "exact content_id"],
              "ranking": ["density * mean L2 quality DESC", "stars DESC", "repo_key", "commit_id", "id"],
              "selection_path": str(selected_path), "evidence_path": str(evidence_path),
              "stats": stats, "by_category": by_category, "file_language_counts": languages,
              "code_policy_summary": policy_summary, "code_policy_length_bins": policy_bins,
              "selected_length_bins": selected_bins,
              "selection_funnel": funnel, "selection_funnel_length_bins": funnel_bins,
              "all_stack_file_license_counts": all_license_groups,
              "selection_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "deduplicated_against_ultra": False, "benchmark_decontaminated": False}
    write_json(destination / (license_policy + ".json"), result)
    print(json.dumps({"license_policy": license_policy, "stats": stats, "selection": str(selected_path)}), flush=True)
    connection.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--deps", default=DEFAULT_DEPS)
    parser.add_argument("--license-policy", choices=["permissive", "all", "both"], default="both")
    parser.add_argument("--min-tokens", type=int, default=8192)
    parser.add_argument("--min-density", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory", default="128GB")
    args = parser.parse_args()
    sys.path.insert(0, args.deps)
    policies = ["permissive", "all"] if args.license_policy == "both" else [args.license_policy]
    for policy in policies:
        select_repos(args.out, policy, args.min_tokens, args.min_density, args.threads, args.memory)


if __name__ == "__main__":
    main()
