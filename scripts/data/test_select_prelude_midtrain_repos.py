"""Check that repository selection uses exact, noninflated file evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("select_repos", Path(__file__).with_name("select_prelude_midtrain_repos.py"))
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)
sys.path.insert(0, selection.DEFAULT_DEPS)


class SelectionTests(unittest.TestCase):
    def test_license_policy_density_and_disjoint_categories(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for directory in ("metrics", "stack_files", "ultra_files"):
                (out / directory).mkdir()
            (out / "input_manifest.json").write_text(json.dumps({"build_id": "fixture"}))
            metrics, files, ultra = [], [], []
            for i,(name,n,tokens,license_type,matches) in enumerate([
                ("a",2,10000,"permissive",2), ("b",5,40000,"permissive",2),
                ("c",5,40000,"permissive",0), ("d",2,15000,"no_license",2)]):
                rid = f"1:{i}"
                metrics.append(dict(id=rid,source="stack-v3-train",subset="repos",file_id=1,row=i,repo_key=name,
                                    repo_path=name,commit_id="commit",stars=0,num_files=n,
                                    permissive_files=n if license_type=="permissive" else 0,
                                    tokens=tokens,permissive_tokens=tokens if license_type=="permissive" else 0,
                                    all_nonvendor_tokens=tokens))
                for f in range(n):
                    row=dict(id=rid,repo_key=name,content_id=f"hash-{f}",file_path=f"{f}.py",
                             license=license_type,language="python",is_vendor=False)
                    files.append(row)
                    if f<matches:
                        ultra.append(dict(row,quality_score=7.0,algo_rel_score=0.9))
            ultra.append(dict(ultra[0]))  # must not inflate the density above 1
            # Wrong repository with the same content hash is not evidence for c.
            ultra.append(dict(ultra[0],repo_key="another-repo"))
            for directory,rows in [("metrics",metrics),("stack_files",files),("ultra_files",ultra)]:
                pq.write_table(pa.Table.from_pylist(rows),out/directory/"part.parquet")
            permissive=selection.select_repos(out,"permissive",threads=1,memory="1GB",require_complete=False)
            all_files=selection.select_repos(out,"all",threads=1,memory="1GB",require_complete=False)
            self.assertEqual(permissive["stats"]["n_repos"],2)
            self.assertEqual(all_files["stats"]["n_repos"],3)
            self.assertEqual(permissive["code_policy_summary"][0]["eligible_tokens"],90000)
            self.assertEqual(all_files["code_policy_summary"][0]["eligible_tokens"],105000)
            self.assertEqual(sum(r["tokens"] for r in all_files["selected_length_bins"]),65000)
            self.assertEqual([r["n_repos"] for r in all_files["selection_funnel"]],[4,4,3,3])
            kept=pq.read_table(permissive["selection_path"]).to_pylist()
            by_repo={r["repo_key"]:r for r in kept}
            self.assertEqual(by_repo["a"]["l2_file_density"],1.0)
            self.assertEqual(by_repo["a"]["mix_category"],"code")
            self.assertEqual(by_repo["b"]["mix_category"],"long_documents")
            self.assertNotIn("c",by_repo)


if __name__=="__main__":
    unittest.main()
