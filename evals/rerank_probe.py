"""临时探针 — 对比 chunk-rerank vs parent-rerank 的分数区分度，判断 rerank 是否有信息增量。

用法: python evals/rerank_probe.py
"""

import asyncio
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from rag.rag_service import init_rag_service
from rag.pipeline.document_repository import get_document_repository
from config.settings import settings

logging.basicConfig(level=logging.ERROR)

CATALOG = {
    "recipe_chunks": {
        "category": ["早餐", "汤类", "主食", "甜品", "饮品", "调料",
                     "半成品加工", "水产", "荤菜", "素菜"],
        "difficulty": ["入门", "简单", "中等", "较难", "困难"],
    }
}

QUERIES = ["茄子菜谱", "推荐几道下饭菜", "回锅肉", "回锅肉的菜谱"]
THRESHOLD = settings.RAG_RERANK_THRESHOLD * 0.9


async def rerank_scores(query, docs_text):
    payload = {"model": settings.SILICONFLOW_MODEL, "query": query, "documents": docs_text}
    headers = {"Authorization": f"Bearer {settings.SILICONFLOW_API_KEY}",
               "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        r = await client.post(settings.SILICONFLOW_BASE_URL, headers=headers,
                              json=payload, timeout=30.0)
        r.raise_for_status()
        return {res["index"]: res["relevance_score"] for res in r.json().get("results", [])}


async def probe(service, repo, query):
    print("\n" + "=" * 80)
    print(f"  QUERY: {query}    (阈值={THRESHOLD:.3f})")
    print("=" * 80)

    rewritten, _ = await service.query_understanding.understand(query, CATALOG)
    docs, rscores = await service.retrieval.hybrid_search(rewritten, top_k=settings.RAG_TOP_K, expr=None)
    if not docs:
        print("  ❌ 召回为空"); return

    # 按 parent 去重，保留每个 parent 的代表 chunk + 最高召回分
    parents = {}  # parent_id -> {name, retr, chunk}
    for d, s in zip(docs, rscores):
        pid = d.metadata.get("parent_id")
        if pid and (pid not in parents or s > parents[pid]["retr"]):
            parents[pid] = {"name": d.metadata.get("dish_name", "?"), "retr": s, "chunk": d.page_content}

    pids = list(parents.keys())
    parent_docs = await repo.get_parent_documents(pids)

    chunk_texts = [parents[p]["chunk"] for p in pids]
    parent_texts = [parent_docs[p].page_content if p in parent_docs else parents[p]["chunk"] for p in pids]

    chunk_rr = await rerank_scores(rewritten, chunk_texts)
    parent_rr = await rerank_scores(rewritten, parent_texts)

    print(f"  改写: {rewritten!r}   候选 {len(pids)} 道菜")
    print(f"\n  {'菜名':<14}{'召回序':>6}{'chunk分':>9}{'父文档分':>10}   gate(chunk/parent)")
    print("  " + "-" * 70)
    rows = sorted(range(len(pids)), key=lambda i: parent_rr.get(i, 0), reverse=True)
    for i in rows:
        nm = parents[pids[i]]["name"]
        c, p = chunk_rr.get(i, 0), parent_rr.get(i, 0)
        cg = "✅" if c >= THRESHOLD else "❌"
        pg = "✅" if p >= THRESHOLD else "❌"
        plen = len(parent_texts[i])
        print(f"  {nm:<14}{parents[pids[i]]['retr']:>6.3f}{c:>9.3f}{p:>10.3f}   {cg}/{pg}  (父{plen}字)")


async def main():
    service = init_rag_service()
    if not service:
        print("❌ init 失败"); sys.exit(1)
    repo = get_document_repository()
    for q in QUERIES:
        try:
            await probe(service, repo, q)
        except Exception as e:
            print(f"  query={q!r} 出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())
