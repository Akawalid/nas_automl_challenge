# Evaluation

For each dataset, the model is initialized and trained from scratch, then your `Trainer.predict()` output is compared against the hidden test labels using [accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).

Each dataset's metadata includes a `benchmark` accuracy -- the score our own reference submission achieved. Your raw accuracy is converted to an adjusted score on a linear scale:

`adj_score = (raw_accuracy - benchmark) * (10 / (100 - benchmark))`

- Matching the benchmark exactly -> 0 points
- 100% accuracy -> +10 points (max)
- Below the benchmark -> negative points, floored at -10
- A timeout, or a crash (e.g. memory), -> -10 for that dataset

Your final score is the **sum** of the adjusted scores across all three datasets (range: -30 to +30).
