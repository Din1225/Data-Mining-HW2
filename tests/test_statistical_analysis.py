"""只測關鍵計算、配對與群組設計；數字均為人工測試資料，不是作業結果。"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import read_score_file
from statistical_analysis import bootstrap_weights, group_paths, holm, weighted_ranking


class StatisticsTests(unittest.TestCase):
    def test_weighted_metrics_against_explicit_resampling_with_ties(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.2, 0.5, 0.5, 0.9, 0.9, 0.2])
        weights = np.array([[1, 2, 1, 3, 0, 2], [0, 1, 2, 0, 3, 1]])
        result = weighted_ranking(y, scores, weights)
        for row, weight in zip(result, weights):
            idx = np.repeat(np.arange(len(y)), weight)
            np.testing.assert_allclose(row, [roc_auc_score(y[idx], scores[idx]),
                                            average_precision_score(y[idx], scores[idx])], atol=1e-14)

    def test_cluster_members_always_resampled_together(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        weights = bootstrap_weights(labels, ["a", "a", "b", "c", "c", "d"], 100, 42)
        np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
        np.testing.assert_array_equal(weights[:, 3], weights[:, 4])
        np.testing.assert_array_equal(weights[:, [0, 2]].sum(axis=1), 2)
        np.testing.assert_array_equal(weights, bootstrap_weights(labels, ["a", "a", "b", "c", "c", "d"], 100, 42))
        with self.assertRaises(ValueError):
            bootstrap_weights(labels, ["a", "a", "b", "a", "c", "d"], 100, 42)

    def test_filename_groups_and_duplicate_union(self):
        paths = ("test/NORMAL/IM-001-0001.jpeg", "test/NORMAL/IM-001-0002.jpeg",
                 "test/NORMAL/IM-002-0001.jpeg", "test/PNEUMONIA/person1_virus_1.jpeg",
                 "test/PNEUMONIA/person1_bacteria_2.jpeg")
        groups = group_paths(paths, [{"paths": paths[1] + ";" + paths[2]}])
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[3], groups[4])

    def test_holm_order_and_bound(self):
        np.testing.assert_allclose(holm([0.04, 0.001, 0.02]), [0.04, 0.003, 0.04])
        np.testing.assert_allclose(holm([0.8, 1.0]), [1.0, 1.0])

    def test_alignment_reorders_and_rejects_missing_duplicate_wrong_labels(self):
        expected = {"val": {"v0": 0, "v1": 1}, "test": {"t0": 0, "t1": 1}}
        rows = [["t1", "test", "PNEUMONIA", 0.8], ["v1", "val", "PNEUMONIA", 0.7],
                ["t0", "test", "NORMAL", 0.2], ["v0", "val", "NORMAL", 0.1]]
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "scores.csv"
            def save(data):
                with p.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["path", "split", "label", "score"])
                    w.writerows(data)
            save(rows)
            val, test = read_score_file(p, expected)
            self.assertEqual(test.paths, ("t0", "t1"))
            np.testing.assert_allclose(val.scores, [0.1, 0.7])
            for bad in (rows[:-1], rows + [rows[0]], [["t1", "test", "NORMAL", 0.8]] + rows[1:]):
                save(bad)
                with self.assertRaises(ValueError):
                    read_score_file(p, expected)


if __name__ == "__main__":
    unittest.main()
