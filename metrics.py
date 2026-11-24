"""Ranking metrics for evaluating a retriever. No dependencies."""
import math


def hit_rate_at_k(ranked, relevant, k):
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / k


def reciprocal_rank(ranked, relevant):
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(runs, qrels, k=3):
    """runs: {query: [doc ids ranked]}, qrels: {query: [relevant ids]}."""
    qs = list(qrels)
    n = len(qs)
    return {
        f"hit@{k}": sum(hit_rate_at_k(runs[q], qrels[q], k) for q in qs) / n,
        f"recall@{k}": sum(recall_at_k(runs[q], qrels[q], k) for q in qs) / n,
        f"precision@{k}": sum(precision_at_k(runs[q], qrels[q], k) for q in qs) / n,
        "MRR": sum(reciprocal_rank(runs[q], qrels[q]) for q in qs) / n,
        f"nDCG@{k}": sum(ndcg_at_k(runs[q], qrels[q], k) for q in qs) / n,
    }
