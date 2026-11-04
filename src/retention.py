"""Backtest of a retention campaign: contact the k highest-scored customers, keep a share of those who would have left."""
import numpy as np


def backtest(churned, value, score, k, save_rate, cost):
    """Net value of contacting the top-k by score. Only customers who really churned yield revenue."""
    idx = np.argsort(-np.asarray(score))[:k]
    return float((np.asarray(churned)[idx] * np.asarray(value)[idx]).sum() * save_rate - k * cost)


def random_backtest(churned, value, k, save_rate, cost, n=500, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([backtest(churned, value, rng.random(len(churned)), k, save_rate, cost) for _ in range(n)]))
