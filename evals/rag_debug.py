"""RAG 模块交互式调试工具 — 逐阶段打印中间结果，定位检索失败的环节。

用法:
    python evals/rag_debug.py

交互命令:
    <直接输入 query>       走完整 pipeline，每一步都打印中间结果
    /raw <query>           只跑 hybrid_search（无任何 LLM 加工），看 Milvus 原始召回
    /noexpr <query>        跑 pipeline 但跳过 LLM metadata filter（隔离 filter 影响）
    /expr <expr>|<query>   手动指定 Milvus expr，看过滤效果
    /topk <n>              修改 top_k（默认 20，影响后续所有 query）
    /q  或  exit            退出

阶段说明:
    [1] Query Rewrite     — LLM 改写后的 query
    [2] Metadata Filter   — LLM 生成的 Milvus expr
    [3a] Hybrid (no filter) — 不加 expr 的纯 hybrid search，看 Milvus 能不能召回
    [3b] Hybrid (+ filter)  — 加 LLM expr 的 hybrid search，对比看 filter 是否过严
    [4] Reranker          — SiliconFlow 重排后的 top docs
    [5] Post-process      — parent_id 去重后的最终菜谱列表
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from rag.rag_service import init_rag_service

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

METADATA_CATALOG = {
    "recipe_chunks": {
        "category": ["早餐", "汤类", "主食", "甜品", "饮品", "调料",
                     "半成品加工", "水产", "荤菜", "素菜"],
        "difficulty": ["入门", "简单", "中等", "较难", "困难"],
    }
}


# ── 终端彩色 ────────────────────────────────────────────────
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def dish_names_of(docs: List[Document]) -> List[str]:
    """按出现顺序去重抽取 dish_name。"""
    seen = set()
    out = []
    for d in docs:
        name = d.metadata.get("dish_name") if d.metadata else None
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def print_section(title: str, color: str = C.CYAN) -> None:
    print(f"\n{color}{C.BOLD}{title}{C.END}")
    print(f"{C.DIM}{'─' * 60}{C.END}")


def print_dishes(docs: List[Document], label: str = "菜谱") -> None:
    """打印 docs 里的菜名（去重）+ chunk 计数。"""
    if not docs:
        print(f"  {C.RED}❌ 返回空{C.END}")
        return
    names = dish_names_of(docs)
    print(f"  共 {len(docs)} chunks → 去重 {len(names)} 道菜:")
    for i, n in enumerate(names, 1):
        score_info = ""
        # 取首次出现这道菜的第一个 chunk 的 score 信息
        for d in docs:
            if d.metadata.get("dish_name") == n:
                rs = d.metadata.get("retrieval_score")
                rr = d.metadata.get("rerank_score")
                bits = []
                if rs is not None:
                    bits.append(f"retr={rs:.3f}")
                if rr is not None:
                    bits.append(f"rerank={rr:.3f}")
                if bits:
                    score_info = f"  {C.DIM}[{' '.join(bits)}]{C.END}"
                break
        print(f"    {i:2d}. {n}{score_info}")


# ── 各阶段调用 ──────────────────────────────────────────────

async def stage_query_rewrite(service, query: str) -> str:
    print_section("[1] Query Rewrite (LLM)", C.BLUE)
    try:
        rewritten = await service.query_rewriter.rewrite_query(query)
        print(f"  原始: {C.YELLOW}{query}{C.END}")
        print(f"  改写: {C.GREEN}{rewritten}{C.END}")
        if rewritten == query:
            print(f"  {C.DIM}（未改写，原样返回）{C.END}")
        return rewritten
    except Exception as e:
        print(f"  {C.RED}❌ 改写失败: {e}{C.END}")
        return query


async def stage_metadata_filter(service, query: str) -> Optional[str]:
    print_section("[2] Metadata Filter (LLM 生成 Milvus expr)", C.BLUE)
    try:
        expr = await service.metadata_filter.build_filter_expression(
            query=query,
            metadata_catalog=METADATA_CATALOG,
        )
        if expr:
            print(f"  生成 expr: {C.YELLOW}{expr}{C.END}")
        else:
            print(f"  {C.DIM}（未生成 expr，将不做硬过滤）{C.END}")
        return expr
    except Exception as e:
        print(f"  {C.RED}❌ 生成失败: {e}{C.END}")
        return None


async def stage_hybrid_search(
    service, query: str, expr: Optional[str], top_k: int, label: str
) -> List[Document]:
    print_section(f"[3] Hybrid Search ({label})", C.BLUE)
    print(f"  query='{query}'  expr={expr}  top_k={top_k}")
    try:
        docs, scores = await service.retrieval.hybrid_search(
            query=query,
            top_k=top_k,
            expr=expr,
        )
        # 把分数写入 metadata 方便后续打印
        for d, s in zip(docs, scores):
            d.metadata["retrieval_score"] = s
        print_dishes(docs)
        return docs
    except Exception as e:
        print(f"  {C.RED}❌ Hybrid search 失败: {e}{C.END}")
        return []


async def stage_rerank(service, query: str, docs: List[Document]) -> List[Document]:
    print_section("[4] Reranker (SiliconFlow cross-encoder)", C.BLUE)
    if not docs:
        print(f"  {C.DIM}（上一步无候选，跳过）{C.END}")
        return []
    try:
        reranked = await service.reranker.rerank(query=query, documents=docs)
        print_dishes(reranked)
        return reranked
    except Exception as e:
        print(f"  {C.RED}❌ Rerank 失败: {e}{C.END}")
        return docs


async def stage_post_process(service, docs: List[Document]) -> List[Document]:
    print_section("[5] Post-process (parent_id 去重 + 取父文档)", C.BLUE)
    if not docs:
        print(f"  {C.DIM}（上一步无候选，跳过）{C.END}")
        return []
    try:
        from rag.pipeline.document_processor import document_processor
        final = await document_processor.post_process_retrieval(docs)
        print_dishes(final)
        return final
    except Exception as e:
        print(f"  {C.RED}❌ 后处理失败: {e}{C.END}")
        return []


# ── 整合流程 ─────────────────────────────────────────────────

async def run_full_diagnosis(service, query: str, top_k: int) -> None:
    """跑完整 pipeline 并打印每一步中间结果。"""
    print(f"\n{C.HEADER}{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.HEADER}{C.BOLD}  完整 Pipeline 诊断: '{query}'{C.END}")
    print(f"{C.HEADER}{C.BOLD}{'═' * 60}{C.END}")

    # [1] Query rewrite
    rewritten = await stage_query_rewrite(service, query)

    # [2] Metadata filter
    expr = await stage_metadata_filter(service, rewritten)

    # [3a] Hybrid search 不加 filter（对照组）
    docs_no_filter = await stage_hybrid_search(
        service, rewritten, expr=None, top_k=top_k,
        label="无 filter，对照组",
    )

    # [3b] Hybrid search 加 LLM 生成的 filter
    if expr:
        docs_with_filter = await stage_hybrid_search(
            service, rewritten, expr=expr, top_k=top_k,
            label="加 LLM filter",
        )
        # 对比信号
        n_no = len(dish_names_of(docs_no_filter))
        n_with = len(dish_names_of(docs_with_filter))
        if n_with < n_no:
            delta = n_no - n_with
            print(f"\n  {C.RED}⚠️  LLM filter 过滤掉了 {delta} 道菜 "
                  f"({n_no} → {n_with}){C.END}")
        elif n_with == 0:
            print(f"\n  {C.RED}⚠️  LLM filter 把候选清零！{C.END}")
    else:
        docs_with_filter = docs_no_filter
        print(f"\n  {C.DIM}（未生成 filter，复用 3a 的结果继续）{C.END}")

    # [4] Rerank
    reranked = await stage_rerank(service, rewritten, docs_with_filter)
    reranked_top = reranked[:10] if reranked else []

    # [5] Post-process
    final = await stage_post_process(service, reranked_top)

    # 最终总结
    print(f"\n{C.HEADER}{C.BOLD}── 最终结果 ──{C.END}")
    final_names = dish_names_of(final)
    if final_names:
        print(f"  {C.GREEN}✅ 返回 {len(final_names)} 道菜: "
              f"{', '.join(final_names)}{C.END}")
    else:
        print(f"  {C.RED}❌ 最终返回为空 — 检查上方哪一步把候选清零了{C.END}")


async def run_raw_hybrid(service, query: str, top_k: int) -> None:
    """只跑 hybrid_search，看 Milvus 原始召回能力（无任何 LLM 加工）。"""
    print(f"\n{C.HEADER}{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.HEADER}{C.BOLD}  Raw Hybrid Search: '{query}'{C.END}")
    print(f"{C.HEADER}{C.BOLD}  （无 rewrite / 无 filter / 无 rerank / 无后处理）{C.END}")
    print(f"{C.HEADER}{C.BOLD}{'═' * 60}{C.END}")
    await stage_hybrid_search(service, query, expr=None, top_k=top_k, label="纯 Milvus 召回")


async def run_no_filter(service, query: str, top_k: int) -> None:
    """跑 pipeline 但跳过 metadata filter（隔离 filter 影响）。"""
    print(f"\n{C.HEADER}{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.HEADER}{C.BOLD}  No-Filter Pipeline: '{query}'{C.END}")
    print(f"{C.HEADER}{C.BOLD}  （rewrite + hybrid + rerank + post-process，跳过 LLM filter）{C.END}")
    print(f"{C.HEADER}{C.BOLD}{'═' * 60}{C.END}")
    rewritten = await stage_query_rewrite(service, query)
    docs = await stage_hybrid_search(service, rewritten, expr=None, top_k=top_k, label="无 filter")
    reranked = await stage_rerank(service, rewritten, docs)
    final = await stage_post_process(service, reranked[:10])
    print(f"\n{C.HEADER}{C.BOLD}── 最终结果 ──{C.END}")
    final_names = dish_names_of(final)
    if final_names:
        print(f"  {C.GREEN}✅ {', '.join(final_names)}{C.END}")
    else:
        print(f"  {C.RED}❌ 返回为空{C.END}")


async def run_with_manual_expr(service, query: str, expr: str, top_k: int) -> None:
    """用用户手动指定的 expr 跑 hybrid search。"""
    print(f"\n{C.HEADER}{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.HEADER}{C.BOLD}  Manual Expr: '{query}'{C.END}")
    print(f"{C.HEADER}{C.BOLD}  expr = {expr}{C.END}")
    print(f"{C.HEADER}{C.BOLD}{'═' * 60}{C.END}")
    await stage_hybrid_search(service, query, expr=expr, top_k=top_k, label="手动 expr")


# ── 主循环 ─────────────────────────────────────────────────

def print_help():
    print(f"\n{C.CYAN}{C.BOLD}可用命令:{C.END}")
    print(f"  {C.YELLOW}<query>{C.END}              完整 pipeline 诊断（每步打印）")
    print(f"  {C.YELLOW}/raw <query>{C.END}         只跑 hybrid search（看 Milvus 原始召回）")
    print(f"  {C.YELLOW}/noexpr <query>{C.END}      跳过 metadata filter 跑 pipeline")
    print(f"  {C.YELLOW}/expr <expr>|<query>{C.END} 手动指定 expr 跑 hybrid search")
    print(f"                       例: /expr category == \"荤菜\"|牛肉的做法")
    print(f"  {C.YELLOW}/topk <n>{C.END}            修改 top_k (当前: {{}})")
    print(f"  {C.YELLOW}/help{C.END} 或 {C.YELLOW}?{C.END}            显示本帮助")
    print(f"  {C.YELLOW}/q{C.END} 或 {C.YELLOW}exit{C.END}            退出\n")


async def main():
    print(f"{C.HEADER}{C.BOLD}RAG 调试工具{C.END}")
    print(f"{C.DIM}初始化 RAG 服务（连接 Milvus + 加载 embedding）...{C.END}")

    service = init_rag_service()
    if service is None:
        print(f"{C.RED}❌ RAG 服务初始化失败 — 检查 Milvus 是否启动、.env 是否配置{C.END}")
        sys.exit(1)

    print(f"{C.GREEN}✅ RAG 服务就绪{C.END}")
    top_k = 20
    print_help()

    while True:
        try:
            raw = input(f"{C.BOLD}query>{C.END} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not raw:
            continue

        if raw in ("/q", "exit", "quit"):
            print("bye")
            return

        if raw in ("/help", "?"):
            print_help()
            continue

        if raw.startswith("/topk "):
            try:
                top_k = int(raw[6:].strip())
                print(f"  top_k = {top_k}")
            except ValueError:
                print(f"  {C.RED}无效的 top_k{C.END}")
            continue

        if raw.startswith("/raw "):
            await run_raw_hybrid(service, raw[5:].strip(), top_k)
            continue

        if raw.startswith("/noexpr "):
            await run_no_filter(service, raw[8:].strip(), top_k)
            continue

        if raw.startswith("/expr "):
            payload = raw[6:].strip()
            if "|" not in payload:
                print(f"  {C.RED}格式: /expr <expr>|<query>{C.END}")
                continue
            expr_str, q_str = payload.split("|", 1)
            await run_with_manual_expr(service, q_str.strip(), expr_str.strip(), top_k)
            continue

        # 默认走完整诊断
        await run_full_diagnosis(service, raw, top_k)


if __name__ == "__main__":
    asyncio.run(main())
