"""RAG 评测指标 — Recall@K, NDCG@K, 延迟统计。"""

import math
import statistics
from typing import Iterable, List


def dedup_keep_order(items: Iterable[str]) -> List[str]:
    """按首次出现顺序去重。"""
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def recall_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    """Recall@K = |retrieved[:k] ∩ expected| / |expected|"""
    if not expected:
        return 0.0
    expected_set = set(expected)
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in expected_set)
    return hits / len(expected_set)


def ndcg_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    """Binary-relevance NDCG@K。"""
    if not expected:
        return 0.0
    expected_set = set(expected)
    top_k = retrieved[:k]

    dcg = 0.0
    for i, r in enumerate(top_k):
        if r in expected_set:
            dcg += 1.0 / math.log2(i + 2)

    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate(retrieved: List[str], expected: List[str], k: int) -> float:
    """命中率：top_k 中是否至少出现一个相关菜（0或1）。"""
    if not expected:
        return 0.0
    expected_set = set(expected)
    return 1.0 if any(r in expected_set for r in retrieved[:k]) else 0.0


def percentile(values: List[float], p: float) -> float:
    """返回 p 百分位数（p ∈ [0, 100]）。values 为空返回 0。"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def latency_stats(latencies_ms: List[float]) -> dict:
    """计算延迟统计 (mean/P50/P95/P99/min/max)。"""
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(latencies_ms),
        "p50": percentile(latencies_ms, 50),
        "p95": percentile(latencies_ms, 95),
        "p99": percentile(latencies_ms, 99),
        "min": min(latencies_ms),
        "max": max(latencies_ms),
    }
