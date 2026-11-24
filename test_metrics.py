import math
from metrics import *


def test_hit_and_recall():
    assert hit_rate_at_k(["a", "b"], ["b"], 2) == 1.0
    assert hit_rate_at_k(["a", "b"], ["b"], 1) == 0.0
    assert recall_at_k(["a", "b", "c"], ["a", "c", "z", "y"], 3) == 0.5


def test_mrr():
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == 1 / 3
    assert reciprocal_rank(["x"], ["a"]) == 0.0


def test_ndcg_perfect_and_worse():
    assert math.isclose(ndcg_at_k(["a", "b"], ["a", "b"], 2), 1.0)
    assert ndcg_at_k(["x", "a"], ["a"], 2) < 1.0


def test_precision():
    assert precision_at_k(["a", "x", "b", "y"], ["a", "b"], 4) == 0.5
