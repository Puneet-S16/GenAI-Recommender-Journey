import numpy as np

def precision_at_k(actual, predicted, k=10):
    predicted = predicted[:k]
    return len(set(predicted) & set(actual)) / k

def recall_at_k(actual, predicted, k=10):
    predicted = predicted[:k]
    return len(set(predicted) & set(actual)) / len(actual)

def map_at_k(actual, predicted, k=10):
    score = 0
    hits = 0

    for i, p in enumerate(predicted[:k]):
        if p in actual:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(actual), k)

def rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted))**2))