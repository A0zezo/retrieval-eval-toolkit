# retrieval-eval-toolkit

Dependency-free ranking metrics for evaluating a retriever or RAG search step: hit@k, recall@k, precision@k, MRR and nDCG@k (binary relevance). `demo.py` scores a keyword-overlap retriever against a random ordering on a 6-document, 5-query labelled set.

## Run

```bash
python demo.py
pip install pytest && pytest
```

## Output (reproduced)

```
keyword_overlap {'hit@3': 1.0, 'recall@3': 1.0, 'precision@3': 0.4, 'MRR': 1.0, 'nDCG@3': 1.0}
random {'hit@3': 0.6, 'recall@3': 0.5, 'precision@3': 0.2, 'MRR': 0.45, 'nDCG@3': 0.349}

$ pytest -q
4 passed in 0.01s
```

## Limits

- The labelled set is tiny and hand-made, so the keyword retriever's perfect score only shows the metrics behave as expected. Use `metrics.evaluate` with your own runs and relevance judgements.
