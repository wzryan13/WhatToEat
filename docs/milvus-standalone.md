# Milvus Standalone Setup

本项目的菜谱检索现在要求使用 `langchain_milvus + BM25BuiltInFunction` 的 hybrid search。
这条链路需要 `Milvus Standalone` 或 `Milvus Distributed`，不能再使用 `Milvus Lite (.db)`。

## 1. 安装 Docker

需要本机安装：

- Docker
- Docker Compose V2

官方安装文档：

- Milvus Docker Compose 安装说明: <https://milvus.io/docs/ja/v2.5.x/install_standalone-docker-compose.md>
- LangChain Milvus 集成说明: <https://docs.langchain.com/oss/python/integrations/vectorstores/milvus>

## 2. 下载官方 Compose 文件并启动

```bash
mkdir -p .milvus/standalone
cd .milvus/standalone
wget https://github.com/milvus-io/milvus/releases/download/v2.5.27/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d
```

启动后默认端口：

- `19530` for Milvus API
- `9091` for web UI / health

## 3. 确认环境变量

项目已默认使用：

```env
MILVUS_URI=http://127.0.0.1:19530
MILVUS_COLLECTION=recipe_chunks
```

## 4. 重建菜谱索引

```bash
python scripts/ingest_recipes.py --data_dir data/HowToCook/dishes --force_rebuild
```

这一步会创建带以下字段的 hybrid collection：

- `text`
- `dense_vector`
- `sparse`

并启用 `BM25BuiltInFunction()`。

## 5. 验证检索

```bash
python - <<'PY'
import asyncio, sys
sys.path.insert(0, '.')
from rag.rag_service import init_rag_service

async def test():
    svc = init_rag_service()
    assert svc, 'RAG 服务初始化失败'
    docs = await svc.search_recipes('番茄炒蛋怎么做')
    assert len(docs) > 0, '检索无结果'
    print(f'成功: 返回 {len(docs)} 条结果')
    for doc in docs[:3]:
        print(f'  {doc.metadata["dish_name"]}')

asyncio.run(test())
PY
```
