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

from config.settings import settings
from rag.embeddings.embedding_factory import get_embedding_model
from rag.vector_stores.vector_store_factory import get_vector_store
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
    embeddings = get_embedding_model(settings.EMBEDDING_MODEL)

    # 4. 确保 Milvus 数据目录存在
    milvus_dir = Path(settings.MILVUS_URI).parent
    milvus_dir.mkdir(parents=True, exist_ok=True)

    # 5. 创建/重建 Milvus 集合并索引
    logger.info(f"索引到 Milvus 集合: {settings.MILVUS_COLLECTION}")
    vectorstore = get_vector_store(
        uri=settings.MILVUS_URI,
        collection_name=settings.MILVUS_COLLECTION,
        embeddings=embeddings,
        chunks=all_chunks,
        force_rebuild=args.force_rebuild,
    )

    logger.info("=" * 50)
    logger.info(f"数据灌入完成!")
    logger.info(f"  菜谱数: {len(recipes)}")
    logger.info(f"  Chunk 数: {len(all_chunks)}")
    logger.info(f"  集合: {settings.MILVUS_COLLECTION}")
    logger.info(f"  存储: {settings.MILVUS_URI}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
