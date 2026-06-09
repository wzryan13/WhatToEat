"""RAG retriever 离线评测脚本。

目标：
1. 评测检索质量：Recall@K / NDCG@K / HitRate@K
2. 支持 baseline vs full pipeline 对比
3. 支持按 query_type 分组统计
4. 支持“按 query_type 差异化启停 metadata filter”的实验策略

默认配置：
- baseline: 纯 hybrid retrieval（无 rewrite / 无 metadata filter / 无 rerank）
- full: 自适应 full pipeline
  - precise query（specific_dish）启用 metadata filter
  - fuzzy query（scene / flavor / ingredient / category / difficulty）关闭 metadata filter

可选模式：
- full_legacy: 历史 full pipeline（所有 query_type 都启用 metadata filter）
- full_no_filter: rewrite + hybrid + rerank + post-process，但完全禁用 metadata filter

用法：
    python evals/run_rag_eval.py
    python evals/run_rag_eval.py --dataset evals/datasets/recipe_eval.json
    python evals/run_rag_eval.py --modes baseline full full_legacy
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from evals.metrics import (
    dedup_keep_order,
    hit_rate,
    latency_stats,
    ndcg_at_k,
    recall_at_k,
)
from rag.pipeline.document_processor import document_processor
from rag.rag_service import init_rag_service

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_eval")
logger.setLevel(logging.INFO)


METADATA_CATALOG = {
    "recipe_chunks": {
        "category": [
            "早餐",
            "汤类",
            "主食",
            "甜品",
            "饮品",
            "调料",
            "半成品加工",
            "水产",
            "荤菜",
            "素菜",
        ],
        "difficulty": ["入门", "简单", "中等", "较难", "困难"],
    }
}


QUERY_TYPE_POLICY = {
    "specific_dish": {
        "enable_metadata_filter": True,
        "reason": "菜名明确，metadata filter 可帮助精确收缩候选集合。",
    },
    "ingredient": {
        "enable_metadata_filter": False,
        "reason": "食材类 query 语义宽，硬过滤容易误杀长尾相关菜。",
    },
    "scene": {
        "enable_metadata_filter": False,
        "reason": "场景类 query 高度模糊，优先保召回，禁用硬过滤。",
    },
    "flavor": {
        "enable_metadata_filter": False,
        "reason": "口味偏好通常无法稳定映射到 category/difficulty 元数据。",
    },
    "category": {
        "enable_metadata_filter": False,
        "reason": "类目词与底层 category 枚举并非一一对应，禁用硬过滤避免坍缩。",
    },
    "difficulty": {
        "enable_metadata_filter": False,
        "reason": "难度类 query 常与早餐/家常/快手等模糊条件混合出现，禁用硬过滤保召回。",
    },
    "unknown": {
        "enable_metadata_filter": False,
        "reason": "未知 query_type 默认保守禁用 metadata filter。",
    },
}


@dataclass(frozen=True)
class ModeConfig:
    name: str
    label: str
    use_query_rewrite: bool
    use_metadata_filter: bool
    adaptive_filter_by_query_type: bool
    use_rerank: bool
    use_post_process: bool
    raw_top_multiplier: int = 5


PROJECT_ROOT = Path(__file__).resolve().parent.parent


MODE_CONFIGS: Dict[str, ModeConfig] = {
    "baseline": ModeConfig(
        name="baseline",
        label="pure hybrid retrieval",
        use_query_rewrite=False,
        use_metadata_filter=False,
        adaptive_filter_by_query_type=False,
        use_rerank=False,
        use_post_process=False,
    ),
    "full": ModeConfig(
        name="full",
        label="adaptive full pipeline",
        use_query_rewrite=True,
        use_metadata_filter=True,
        adaptive_filter_by_query_type=True,
        use_rerank=True,
        use_post_process=True,
    ),
    "full_legacy": ModeConfig(
        name="full_legacy",
        label="legacy full pipeline",
        use_query_rewrite=True,
        use_metadata_filter=True,
        adaptive_filter_by_query_type=False,
        use_rerank=True,
        use_post_process=True,
    ),
    "full_no_filter": ModeConfig(
        name="full_no_filter",
        label="full pipeline without metadata filter",
        use_query_rewrite=True,
        use_metadata_filter=False,
        adaptive_filter_by_query_type=False,
        use_rerank=True,
        use_post_process=True,
    ),
}


def load_dataset(path: Path) -> List[dict]:
    """同时支持 JSON 数组和 JSONL。"""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON dataset 必须是数组格式")
        return data

    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def resolve_project_path(path_str: str) -> Path:
    """优先按传入路径解析；若不存在，则回退到项目根下解析。"""
    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    project_candidate = PROJECT_ROOT / path_str
    if project_candidate.exists():
        return project_candidate

    return candidate


def extract_dish_names(docs_or_dicts: List[Any]) -> List[str]:
    """从 Document / dict 列表中抽取去重后的 dish_name。"""
    names: List[str] = []
    for item in docs_or_dicts:
        meta = item.metadata if hasattr(item, "metadata") else item.get("metadata", {})
        name = meta.get("dish_name") if meta else None
        if name:
            names.append(name)
    return dedup_keep_order(names)


def build_summary(per_query: List[dict], top_k: int) -> Dict[str, Any]:
    def avg(key: str) -> float:
        return sum(item[key] for item in per_query) / len(per_query) if per_query else 0.0

    latencies = [item["latency_ms"] for item in per_query]
    return {
        "count": len(per_query),
        "recall_at_5": avg("recall_at_5"),
        "recall_at_k": avg("recall_at_k"),
        "ndcg_at_k": avg("ndcg_at_k"),
        "hit_at_k": avg("hit_at_k"),
        "latency": latency_stats(latencies),
        "top_k": top_k,
    }


def build_type_summary(per_query: List[dict], top_k: int) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for item in per_query:
        grouped[item["query_type"]].append(item)

    return {
        query_type: build_summary(rows, top_k)
        for query_type, rows in grouped.items()
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_improvement(base: float, new: float) -> str:
    if base == 0:
        return "+∞" if new > 0 else "—"
    delta = (new - base) / base * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def should_enable_metadata_filter(mode: ModeConfig, query_type: str) -> bool:
    if not mode.use_metadata_filter:
        return False
    if not mode.adaptive_filter_by_query_type:
        return True
    policy = QUERY_TYPE_POLICY.get(query_type, QUERY_TYPE_POLICY["unknown"])
    return bool(policy["enable_metadata_filter"])


async def execute_mode(
    service,
    item: dict,
    mode: ModeConfig,
    top_k: int,
) -> Dict[str, Any]:
    query = item["query"]
    query_type = item.get("query_type", "unknown")
    expected = item["expected_dishes"]

    t0 = time.perf_counter()

    rewritten_query = query
    if mode.use_query_rewrite:
        rewritten_query = await service.query_rewriter.rewrite_query(query)

    applied_filter = should_enable_metadata_filter(mode, query_type)
    filter_expr: Optional[str] = None
    if applied_filter:
        filter_expr = await service.metadata_filter.build_filter_expression(
            query=rewritten_query,
            metadata_catalog=METADATA_CATALOG,
        )

    raw_top_k = settings.RAG_TOP_K if (mode.use_rerank or mode.use_post_process) else top_k * mode.raw_top_multiplier
    docs, scores = await service.retrieval.hybrid_search(
        query=rewritten_query,
        top_k=raw_top_k,
        expr=filter_expr,
    )

    for doc, score in zip(docs, scores):
        doc.metadata["retrieval_score"] = score

    if mode.use_rerank and docs:
        docs = await service.reranker.rerank(query=rewritten_query, documents=docs)
        docs = docs[: settings.RAG_RERANK_TOP_K]

    if mode.use_post_process and docs:
        docs = await document_processor.post_process_retrieval(docs)

    latency_ms = (time.perf_counter() - t0) * 1000
    retrieved = extract_dish_names(docs)[:top_k]
    expected_set = set(expected)
    hits = [name for name in retrieved if name in expected_set]

    return {
        "query": query,
        "query_type": query_type,
        "expected": expected,
        "retrieved": retrieved,
        "hits": hits,
        "rewritten_query": rewritten_query,
        "applied_metadata_filter": applied_filter,
        "filter_expr": filter_expr,
        "strategy_note": QUERY_TYPE_POLICY.get(query_type, QUERY_TYPE_POLICY["unknown"])["reason"],
        "latency_ms": latency_ms,
    }


async def evaluate_mode(
    service,
    dataset: List[dict],
    mode: ModeConfig,
    top_k: int,
) -> Dict[str, Any]:
    per_query: List[dict] = []

    for item in dataset:
        qid = item["id"]
        try:
            result = await execute_mode(service, item, mode, top_k)
        except Exception as exc:
            logger.error("[%s] %s 检索失败: %s", mode.name, qid, exc)
            result = {
                "query": item["query"],
                "query_type": item.get("query_type", "unknown"),
                "expected": item["expected_dishes"],
                "retrieved": [],
                "hits": [],
                "rewritten_query": item["query"],
                "applied_metadata_filter": False,
                "filter_expr": None,
                "strategy_note": "pipeline 执行异常，结果记为空。",
                "latency_ms": 0.0,
            }

        expected = result["expected"]
        per_query_row = {
            "id": qid,
            "query": result["query"],
            "query_type": result["query_type"],
            "expected": expected,
            "expected_count": len(expected),
            "retrieved": result["retrieved"],
            "hits": result["hits"],
            "rewritten_query": result["rewritten_query"],
            "applied_metadata_filter": result["applied_metadata_filter"],
            "filter_expr": result["filter_expr"],
            "strategy_note": result["strategy_note"],
            "recall_at_5": recall_at_k(result["retrieved"], expected, 5),
            "recall_at_k": recall_at_k(result["retrieved"], expected, top_k),
            "ndcg_at_k": ndcg_at_k(result["retrieved"], expected, top_k),
            "hit_at_k": hit_rate(result["retrieved"], expected, top_k),
            "latency_ms": result["latency_ms"],
        }
        per_query.append(per_query_row)

        print(
            f"  [{mode.name}] {qid} {result['query'][:24]:<24} "
            f"R@5={per_query_row['recall_at_5']:.2f} "
            f"R@{top_k}={per_query_row['recall_at_k']:.2f} "
            f"NDCG@{top_k}={per_query_row['ndcg_at_k']:.2f} "
            f"Hit@{top_k}={per_query_row['hit_at_k']:.2f} "
            f"{per_query_row['latency_ms']:6.0f}ms "
            f"filter={'Y' if per_query_row['applied_metadata_filter'] else 'N'}"
        )

    return {
        "name": mode.name,
        "config": asdict(mode),
        "per_query": per_query,
        "overall": build_summary(per_query, top_k),
        "by_type": build_type_summary(per_query, top_k),
    }


def write_markdown_report(
    report_path: Path,
    dataset_path: Path,
    top_k: int,
    mode_names: List[str],
    results: Dict[str, Dict[str, Any]],
) -> None:
    baseline_name = "baseline" if "baseline" in results else mode_names[0]
    full_name = "full" if "full" in results else mode_names[1]
    base = results[baseline_name]
    full = results[full_name]

    lines: List[str] = []
    lines.append("# RAG 检索评测报告")
    lines.append("")
    lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **数据集**: `{dataset_path}` ({len(base['per_query'])} queries)")
    lines.append(f"- **Top-K**: {top_k}")
    lines.append("- **评测对象**: RAG retriever 子系统（绕过 intent_parser / memory / graph）")
    lines.append("- **说明**: 评测脚本直接调用 retrieval / rewrite / metadata filter / rerank 组件，关闭缓存干扰，强调可复现实验。")
    lines.append("")

    lines.append("## Query 类型策略")
    lines.append("")
    lines.append("| Query 类型 | Metadata Filter 策略 | 说明 |")
    lines.append("|---|---|---|")
    for query_type, policy in QUERY_TYPE_POLICY.items():
        enabled = "启用" if policy["enable_metadata_filter"] else "关闭"
        lines.append(f"| {query_type} | {enabled} | {policy['reason']} |")
    lines.append("")

    lines.append("## 评测模式")
    lines.append("")
    lines.append("| Mode | Query Rewrite | Metadata Filter | Adaptive by query_type | Rerank | Post-process |")
    lines.append("|---|---|---|---|---|---|")
    for mode_name in mode_names:
        config = MODE_CONFIGS[mode_name]
        lines.append(
            f"| {mode_name} | "
            f"{'✅' if config.use_query_rewrite else '❌'} | "
            f"{'✅' if config.use_metadata_filter else '❌'} | "
            f"{'✅' if config.adaptive_filter_by_query_type else '❌'} | "
            f"{'✅' if config.use_rerank else '❌'} | "
            f"{'✅' if config.use_post_process else '❌'} |"
        )
    lines.append("")

    lines.append("## 整体指标")
    lines.append("")
    lines.append(f"| Mode | Recall@5 | Recall@{top_k} | NDCG@{top_k} | HitRate@{top_k} | P50(ms) | P95(ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for mode_name in mode_names:
        overall = results[mode_name]["overall"]
        lines.append(
            f"| {mode_name} | "
            f"{format_pct(overall['recall_at_5'])} | "
            f"{format_pct(overall['recall_at_k'])} | "
            f"{overall['ndcg_at_k']:.3f} | "
            f"{format_pct(overall['hit_at_k'])} | "
            f"{overall['latency']['p50']:.0f} | "
            f"{overall['latency']['p95']:.0f} |"
        )
    lines.append("")

    lines.append(f"## Baseline vs Full（{baseline_name} vs {full_name}）")
    lines.append("")
    bo = base["overall"]
    fo = full["overall"]
    lines.append("| 指标 | Baseline | Full | 相对提升 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Recall@5 | {format_pct(bo['recall_at_5'])} | {format_pct(fo['recall_at_5'])} | {format_improvement(bo['recall_at_5'], fo['recall_at_5'])} |")
    lines.append(f"| Recall@{top_k} | {format_pct(bo['recall_at_k'])} | {format_pct(fo['recall_at_k'])} | {format_improvement(bo['recall_at_k'], fo['recall_at_k'])} |")
    lines.append(f"| NDCG@{top_k} | {bo['ndcg_at_k']:.3f} | {fo['ndcg_at_k']:.3f} | {format_improvement(bo['ndcg_at_k'], fo['ndcg_at_k'])} |")
    lines.append(f"| HitRate@{top_k} | {format_pct(bo['hit_at_k'])} | {format_pct(fo['hit_at_k'])} | {format_improvement(bo['hit_at_k'], fo['hit_at_k'])} |")
    lines.append("")

    lines.append("## 按 Query 类型分组")
    lines.append("")
    lines.append(
        f"| Query 类型 | 数量 | "
        f"Baseline Recall@{top_k} | Full Recall@{top_k} | Recall 提升 | "
        f"Baseline NDCG@{top_k} | Full NDCG@{top_k} | "
        f"Baseline Hit@{top_k} | Full Hit@{top_k} |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    all_types = sorted(set(base["by_type"].keys()) | set(full["by_type"].keys()))
    for query_type in all_types:
        b = base["by_type"].get(query_type, build_summary([], top_k))
        f = full["by_type"].get(query_type, build_summary([], top_k))
        lines.append(
            f"| {query_type} | {b['count'] or f['count']} | "
            f"{format_pct(b['recall_at_k'])} | {format_pct(f['recall_at_k'])} | {format_improvement(b['recall_at_k'], f['recall_at_k'])} | "
            f"{b['ndcg_at_k']:.3f} | {f['ndcg_at_k']:.3f} | "
            f"{format_pct(b['hit_at_k'])} | {format_pct(f['hit_at_k'])} |"
        )
    lines.append("")

    lines.append("## Per-Query 明细")
    lines.append("")
    lines.append(
        f"| ID | Query | 类型 | Baseline Recall@{top_k} | Full Recall@{top_k} | "
        f"Full NDCG@{top_k} | Full Hit@{top_k} | Full Filter | Full 延迟 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    base_by_id = {row["id"]: row for row in base["per_query"]}
    full_by_id = {row["id"]: row for row in full["per_query"]}
    for qid in sorted(full_by_id.keys()):
        b = base_by_id[qid]
        f = full_by_id[qid]
        lines.append(
            f"| {qid} | {f['query']} | {f['query_type']} | "
            f"{format_pct(b['recall_at_k'])} | {format_pct(f['recall_at_k'])} | "
            f"{f['ndcg_at_k']:.3f} | {format_pct(f['hit_at_k'])} | "
            f"{'ON' if f['applied_metadata_filter'] else 'OFF'} | "
            f"{f['latency_ms']:.0f}ms |"
        )
    lines.append("")

    failed = [row for row in full["per_query"] if row["recall_at_k"] == 0.0]
    if failed:
        lines.append(f"## Full 失败 Case（Recall@{top_k}=0）")
        lines.append("")
        for row in failed:
            expected_str = ", ".join(row["expected"])
            retrieved_str = ", ".join(row["retrieved"][:top_k]) if row["retrieved"] else "[]"
            lines.append(f"### {row['id']}: {row['query']}")
            lines.append(f"- query_type: {row['query_type']}")
            lines.append(f"- metadata_filter: {'ON' if row['applied_metadata_filter'] else 'OFF'}")
            lines.append(f"- expected: {expected_str}")
            lines.append(f"- retrieved: {retrieved_str}")
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retriever 离线评测")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evals/datasets/recipe_eval.json",
        help="评测集路径，支持 .json / .jsonl",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="评测 top_k，默认 10",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evals/reports",
        help="报告输出目录",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "full"],
        choices=sorted(MODE_CONFIGS.keys()),
        help="要运行的评测模式列表",
    )
    args = parser.parse_args()

    dataset_path = resolve_project_path(args.dataset)
    if not dataset_path.exists():
        logger.error("数据集不存在: %s", dataset_path)
        sys.exit(1)

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("初始化 RAG 服务...")
    service = init_rag_service()
    if service is None:
        logger.error("RAG 服务初始化失败，请检查 Milvus / 环境变量")
        sys.exit(1)

    dataset = load_dataset(dataset_path)
    logger.info("加载评测集: %d queries", len(dataset))

    results: Dict[str, Dict[str, Any]] = {}
    for mode_name in args.modes:
        mode = MODE_CONFIGS[mode_name]
        print("\n" + "=" * 88)
        print(f"{mode.name}: {mode.label}")
        print("=" * 88)
        results[mode_name] = await evaluate_mode(
            service=service,
            dataset=dataset,
            mode=mode,
            top_k=args.top_k,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"report_{timestamp}.md"
    json_path = output_dir / f"report_{timestamp}.json"

    write_markdown_report(
        report_path=report_path,
        dataset_path=dataset_path,
        top_k=args.top_k,
        mode_names=args.modes,
        results=results,
    )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "timestamp": timestamp,
                "dataset": str(dataset_path),
                "top_k": args.top_k,
                "query_type_policy": QUERY_TYPE_POLICY,
                "modes": args.modes,
                "results": results,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 88)
    print("评测完成")
    print("=" * 88)
    print(f"{'Mode':<16} {'Recall@5':<12} {f'Recall@{args.top_k}':<12} {f'NDCG@{args.top_k}':<12} {f'Hit@{args.top_k}':<12} {'P50(ms)':<10}")
    for mode_name in args.modes:
        overall = results[mode_name]["overall"]
        print(
            f"{mode_name:<16} "
            f"{format_pct(overall['recall_at_5']):<12} "
            f"{format_pct(overall['recall_at_k']):<12} "
            f"{overall['ndcg_at_k']:.3f}{'':<7} "
            f"{format_pct(overall['hit_at_k']):<12} "
            f"{overall['latency']['p50']:.0f}"
        )

    if "baseline" in results and "full" in results:
        bo = results["baseline"]["overall"]
        fo = results["full"]["overall"]
        print("")
        print("Baseline vs Full")
        print(f"  Recall@5   : {format_pct(bo['recall_at_5'])} -> {format_pct(fo['recall_at_5'])} ({format_improvement(bo['recall_at_5'], fo['recall_at_5'])})")
        print(f"  Recall@{args.top_k:<4}: {format_pct(bo['recall_at_k'])} -> {format_pct(fo['recall_at_k'])} ({format_improvement(bo['recall_at_k'], fo['recall_at_k'])})")
        print(f"  NDCG@{args.top_k:<6}: {bo['ndcg_at_k']:.3f} -> {fo['ndcg_at_k']:.3f} ({format_improvement(bo['ndcg_at_k'], fo['ndcg_at_k'])})")
        print(f"  Hit@{args.top_k:<7}: {format_pct(bo['hit_at_k'])} -> {format_pct(fo['hit_at_k'])} ({format_improvement(bo['hit_at_k'], fo['hit_at_k'])})")

    print("")
    print(f"Markdown 报告: {report_path}")
    print(f"JSON 报告:     {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
