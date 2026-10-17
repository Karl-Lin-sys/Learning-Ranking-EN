import json
import math
from http.server import HTTPServer
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import torch
from travel_rank.core import *
from travel_rank.demo import synthetic_queries, search, catalog
from travel_rank.model import fit, predict, ranking_loss
from travel_rank.api import Handler
from travel_rank.operations import analyze_ab, log_summary


class RetrievalTests(unittest.TestCase):
    def test_hard_filters(self):
        result = search("親子 飯店", [.9, .1, .4])
        self.assertEqual(set(result["candidates"]), {"p1", "p2", "p5"})
        self.assertNotIn("p3", result["results"])
        self.assertEqual(search("親子", [.9, .1, .4], budget=0)["results"], [])

    def test_rrf_deduplicates_routes(self):
        self.assertEqual(rrf([["a", "a", "b"], ["b", "c"]]), ["b", "a", "c"])

    def test_metrics_pool_denominator(self):
        self.assertAlmostEqual(metrics(["a"], {"a": 1, "b": 1})["recall"], .5)
        self.assertLess(metrics(["a"], {"a": 1, "b": 3})["ndcg"], 1)
        self.assertEqual(metrics([], {"a": 1})["ndcg"], 0)
        self.assertEqual(metrics(["a"], {"a": 0})["ndcg"], 0)

    def test_reject_unjudged_and_duplicates(self):
        for ranking in (["x"], ["a", "a"]):
            with self.assertRaises(ValueError):
                metrics(ranking, {"a": 1})

    def test_vector_invalid(self):
        for vector in ([0., 0., 0.], [1., 2.], [math.nan, 1., 2.]):
            with self.assertRaises(ValueError):
                dense_search(vector, catalog(), 3)

    def test_empty_lexical(self):
        self.assertEqual(BM25(catalog()).search("不存在zzzz", {"p1"}, 3), [])


class IntegrityTests(unittest.TestCase):
    def test_time_split(self):
        rows = synthetic_queries(12)
        parts = temporal_split(rows, 40, 80, 120)
        self.assertEqual([len(p) for p in parts], [4, 4, 4])
        rows[3]["session"] = rows[4]["session"]
        parts = temporal_split(rows, 40, 80, 120)
        self.assertEqual([len(p) for p in parts], [3, 3, 4])

    def test_immature_labels(self):
        rows = synthetic_queries(12)
        rows[3]["label_time"] = 41
        self.assertEqual(len(temporal_split(rows, 40, 80, 120)[0]), 3)

    def test_future_features_and_duplicates(self):
        rows = synthetic_queries(12)
        rows[0]["feature_time"] = 99
        with self.assertRaises(ValueError):
            temporal_split(rows, 40, 80, 120)
        rows = synthetic_queries(12)
        with self.assertRaises(ValueError):
            temporal_split(rows + [rows[0]], 40, 80, 120)

    def test_ips_known_example_and_support(self):
        self.assertAlmostEqual(ips_mean([1, 0], [.5, .5])["ips"], 1)
        self.assertEqual(ips_mean([1], [.01])["clipped_fraction"], 1)
        with self.assertRaises(ValueError):
            ips_mean([1], [0])

    def test_bootstrap_and_assignment(self):
        self.assertEqual(paired_bootstrap([.2, .2], 100)["ci95"], [.2, .2])
        self.assertEqual(experiment_arm("u"), experiment_arm("u"))
        self.assertEqual({experiment_arm(str(i)) for i in range(100)}, {"A", "B"})


class ModelTests(unittest.TestCase):
    def test_losses_prefer_correct_order(self):
        y = torch.tensor([[3., 0.]])
        for method in ("pointwise", "pairwise", "listwise"):
            good = ranking_loss(torch.tensor([[2., -2.]]), y, method)
            bad = ranking_loss(torch.tensor([[-2., 2.]]), y, method)
            self.assertLess(good, bad)

    def test_no_pairs_zero_gradient(self):
        s = torch.tensor([[1., 2.]], requires_grad=True)
        loss = ranking_loss(s, torch.zeros_like(s), "pairwise")
        loss.backward()
        self.assertEqual(s.grad.tolist(), [[0., 0.]])

    def test_training_and_inference(self):
        torch.set_num_threads(1)
        train, valid, _ = temporal_split(synthetic_queries(50), 300, 400, 500)
        model, mean, scale, history = fit(train, valid, epochs=4)
        output = predict(model, mean, scale, [valid[0]["x"]])
        self.assertEqual(len(output[0]), 12)
        self.assertTrue(all(math.isfinite(x) for x in output[0]))
        self.assertFalse(model.training)
        self.assertEqual(mean.tolist(), torch.tensor([r["x"] for r in train]).mean((0, 1)).tolist())


class OperationsTests(unittest.TestCase):
    def test_ab_balanced_and_duplicate(self):
        rows = [{"unit": str(i), "arm": "A" if i < 20 else "B", "converted": i % 2, "matured": True} for i in range(40)]
        result = analyze_ab(rows, 100)
        self.assertEqual(result["absolute_delta"], 0)
        self.assertEqual(result["srm_p_approx"], 1)
        with self.assertRaises(ValueError):
            analyze_ab(rows + [rows[0]])
        rows[0]["matured"] = False
        with self.assertRaises(ValueError):
            analyze_ab(rows)

    def test_monitor(self):
        result = log_summary([{"latency_ms": i, "results": i % 2, "error": False} for i in range(1, 21)])
        self.assertEqual(result["p95_ms"], 19)
        self.assertEqual(result["zero_result_rate"], .5)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def test_health_and_search(self):
        with urlopen(self.url + "/health", timeout=5) as response:
            self.assertTrue(json.load(response)["synthetic"])
        req = Request(self.url + "/search", data=json.dumps({"query": "親子", "vector": [.9, .1, .4]}).encode(), method="POST")
        with urlopen(req, timeout=5) as response:
            self.assertEqual(set(json.load(response)["results"]), {"p1", "p2", "p5"})

    def test_malformed_and_oversize(self):
        for body in (b"not json", b"x" * 4097, b'{"query":"a","vector":[0,0,0]}'):
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(self.url + "/search", data=body), timeout=5)
            self.assertEqual(error.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
