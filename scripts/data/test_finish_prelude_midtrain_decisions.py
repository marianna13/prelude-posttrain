"""Parallel decision completion must reproduce the frozen reducer's manifest."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import reduce_prelude_midtrain as reducer
from finish_prelude_midtrain_decisions import finish
import test_curate_prelude_midtrain as fixtures


class CompletionParityTest(unittest.TestCase):
    def test_parallel_completion_matches_original_manifest(self):
        original = reducer.decide

        def decide_and_compare(args):
            original(args)
            path = Path(args.out) / 'reduction/decision_manifest.json'
            expected = path.read_bytes()
            self.assertEqual(finish(args.out, workers=2), json.loads(expected))
            self.assertEqual(path.read_bytes(), expected)

        with patch.object(reducer, 'decide', side_effect=decide_and_compare):
            fixture = fixtures.GlobalReductionTest('test_actual_duckdb_reservation_code_coverage_aliases_and_boilerplate')
            fixture.test_actual_duckdb_reservation_code_coverage_aliases_and_boilerplate()


if __name__ == '__main__':
    unittest.main()
