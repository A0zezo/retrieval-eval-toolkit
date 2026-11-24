"""Compares two toy retrievers (keyword overlap vs. shuffled baseline) on a tiny labelled set."""
import json, random
from metrics import evaluate

data = json.load(open("data.json"))
docs, qrels = data["docs"], data["qrels"]


def keyword_rank(q):
    qw = set(q.lower().split())
    return sorted(docs, key=lambda d: -len(qw & set(docs[d].lower().split())))


def random_rank(q):
    ids = list(docs)
    random.Random(0).shuffle(ids)
    return ids


for name, fn in [("keyword_overlap", keyword_rank), ("random", random_rank)]:
    scores = evaluate({q: fn(q) for q in qrels}, qrels, k=3)
    print(name, {m: round(v, 3) for m, v in scores.items()})
