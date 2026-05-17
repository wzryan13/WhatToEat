# scripts/ingest_recipes.py
"""
菜谱数据灌入脚本 — 解析 HowToCook Markdown 文件并索引到 Milvus。

使用方法:
    python scripts/ingest_recipes.py --data_dir /path/to/HowToCook/dishes

目录结构示例（HowToCook）:
    dishes/
    ├── home-cooking/        # 家常菜
    │   ├── 番茄炒蛋.md
    │   └── 红烧肉.md
    ├── breakfast/           # 早餐
    │   └── 煎蛋.md
    └── soup/                # 汤
        └── 紫菜蛋花汤.md
"""

import argparse
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

from config.settings import settings
from rag.embeddings.embedding_factory import get_embedding_model
from rag.pipeline.document_processor import document_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 目录名 → 中文分类映射
CATEGORY_MAP = {
    "home-cooking": "家常菜",
    "breakfast": "早餐",
    "soup": "汤类",
    "staple": "主食",
    "dessert": "甜品",
    "drink": "饮品",
    "condiment": "调料",
    "semi-finished": "半成品加工",
    "aquatic": "水产",
    "meat_dish": "荤菜",
    "vegetable_dish": "素菜",
}

# 难度星级映射
DIFFICULTY_MAP = {
    1: "入门",
    2: "简单",
    3: "中等",
    4: "较难",
    5: "困难",
}


def parse_recipe_file(file_path: Path, category: str) -> Tuple[str, Dict]:
    """
    解析单个菜谱 Markdown 文件。

    Returns:
        (content, metadata) 元组
    """
    content = file_path.read_text(encoding="utf-8")

    # 提取菜名（从文件名，去掉 .md 后缀和"的做法"后缀）
    dish_name = file_path.stem
    dish_name = re.sub(r"的做法$", "", dish_name)

    # 提取难度（从"预估烹饪难度：★★★★"格式）
    difficulty = "未知"
    diff_match = re.search(r"预估烹饪难度[：:]\s*([★☆]+)", content)
    if diff_match:
        stars = diff_match.group(1).count("★")
        difficulty = DIFFICULTY_MAP.get(stars, f"{stars}星")

    metadata = {
        "source": str(file_path),
        "parent_id": "",  # 将在 create_chunks 中设置
        "dish_name": dish_name,
        "category": category,
        "difficulty": difficulty,
        "is_dish_index": False,
        "data_source": "howtocook",
        "user_id": "system",
        "source_type": "markdown",
    }

    return content, metadata


def scan_recipe_directory(data_dir: Path) -> List[Tuple[str, Dict]]:
    """
    扫描菜谱目录，返回 (content, metadata) 列表。
    """
    recipes = []

    for item in data_dir.iterdir():
        if item.is_dir():
            # 目录名作为分类
            category = CATEGORY_MAP.get(item.name, item.name)

            for md_file in item.glob("*.md"):
                if md_file.name.startswith(".") or md_file.name == "README.md":
                    continue
                try:
                    content, metadata = parse_recipe_file(md_file, category)
                    recipes.append((content, metadata))
                except Exception as e:
                    logger.warning(f"解析文件失败 {md_file}: {e}")
        elif item.suffix == ".md" and item.name != "README.md":
            # 根目录下的 md 文件
            try:
                content, metadata = parse_recipe_file(item, "未分类")
                recipes.append((content, metadata))
            except Exception as e:
                logger.warning(f"解析文件失败 {item}: {e}")

    return recipes


def main():
    parser = argparse.ArgumentParser(description="菜谱数据灌入 Milvus")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="HowToCook 菜谱目录路径（如 /path/to/HowToCook/dishes）",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="强制重建 Milvus 集合（删除已有数据）",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"目录不存在: {data_dir}")
        sys.exit(1)

    # 1. 扫描菜谱文件
    logger.info(f"扫描菜谱目录: {data_dir}")
    recipes = scan_recipe_directory(data_dir)
    logger.info(f"找到 {len(recipes)} 个菜谱文件")

    if not recipes:
        logger.error("未找到任何菜谱文件")
        sys.exit(1)

    # 2. 分块
    logger.info("开始分块...")
    all_chunks: List[Document] = []
    for content, metadata in recipes:
        doc_id = str(uuid.uuid4())
        chunks = document_processor.create_chunks(
            doc_id=doc_id,
            content=content,
            metadata=metadata,
        )
        all_chunks.extend(chunks)

    logger.info(f"共生成 {len(all_chunks)} 个 chunks")

    # 3. 初始化 embedding 模型
    logger.info(f"加载 embedding 模型: {settings.EMBEDDING_MODEL}")
    embedding_model = get_embedding_model(settings.EMBEDDING_MODEL)

    # 4. 确保 Milvus 数据目录存在
    milvus_dir = Path(settings.MILVUS_URI).parent
    milvus_dir.mkdir(parents=True, exist_ok=True)

    # 5. 生成 embeddings
    logger.info("生成 embeddings（可能需要几分钟）...")
    texts = [chunk.page_content for chunk in all_chunks]
    # 分批 embed 避免内存问题
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_emb = embedding_model.embed_documents(batch)
        all_embeddings.extend(batch_emb)
        if (i // batch_size) % 5 == 0:
            logger.info(f"  已处理 {min(i + batch_size, len(texts))}/{len(texts)} 个文本")

    dim = len(all_embeddings[0])
    logger.info(f"Embedding 维度: {dim}, 共 {len(all_embeddings)} 个向量")

    # 6. 使用 MilvusClient 直接创建集合并插入
    uri = settings.MILVUS_URI
    collection_name = settings.MILVUS_COLLECTION

    if args.force_rebuild and os.path.exists(uri):
        import shutil
        logger.warning(f"强制重建: 删除数据库 {uri}")
        if os.path.isdir(uri):
            shutil.rmtree(uri)
        else:
            os.remove(uri)

    logger.info(f"连接 Milvus Lite: {uri}")
    client = MilvusClient(uri=uri)

    # 如果集合已存在且 force_rebuild，先删除
    if client.has_collection(collection_name):
        if args.force_rebuild:
            client.drop_collection(collection_name)
            logger.info(f"已删除旧集合: {collection_name}")
        else:
            logger.info(f"集合已存在: {collection_name}，追加数据")

    # 创建集合 schema
    if not client.has_collection(collection_name):
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
        # 元数据字段
        schema.add_field("category", DataType.VARCHAR, max_length=128)
        schema.add_field("difficulty", DataType.VARCHAR, max_length=64)
        schema.add_field("dish_name", DataType.VARCHAR, max_length=256)
        schema.add_field("user_id", DataType.VARCHAR, max_length=64)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
        schema.add_field("data_source", DataType.VARCHAR, max_length=64)
        schema.add_field("source_type", DataType.VARCHAR, max_length=64)

        # 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
        )

        # 创建向量索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="FLAT",
            metric_type="COSINE",
        )
        client.create_index(collection_name, index_params)
        logger.info(f"集合 '{collection_name}' 创建成功")

    # 7. 批量插入数据
    logger.info("开始批量插入...")
    insert_batch_size = 100
    total_inserted = 0

    for i in range(0, len(all_chunks), insert_batch_size):
        batch_chunks = all_chunks[i:i + insert_batch_size]
        batch_vectors = all_embeddings[i:i + insert_batch_size]

        data = []
        for chunk, vector in zip(batch_chunks, batch_vectors):
            row = {
                "text": chunk.page_content[:65535],
                "dense_vector": vector,
                "category": chunk.metadata.get("category", ""),
                "difficulty": chunk.metadata.get("difficulty", ""),
                "dish_name": chunk.metadata.get("dish_name", ""),
                "user_id": chunk.metadata.get("user_id", ""),
                "parent_id": chunk.metadata.get("parent_id", ""),
                "data_source": chunk.metadata.get("data_source", ""),
                "source_type": chunk.metadata.get("source_type", ""),
            }
            data.append(row)

        client.insert(collection_name=collection_name, data=data)
        total_inserted += len(data)
        if (i // insert_batch_size) % 3 == 0:
            logger.info(f"  已插入 {total_inserted}/{len(all_chunks)}")

    # 8. 加载集合到内存（供搜索）
    client.load_collection(collection_name)

    logger.info("=" * 50)
    logger.info(f"数据灌入完成!")
    logger.info(f"  菜谱数: {len(recipes)}")
    logger.info(f"  Chunk 数: {len(all_chunks)}")
    logger.info(f"  已插入: {total_inserted}")
    logger.info(f"  集合: {collection_name}")
    logger.info(f"  存储: {uri}")
    logger.info("=" * 50)

    client.close()


if __name__ == "__main__":
    main()
