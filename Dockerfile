# WhatToEat 应用镜像 — FastAPI 后端
FROM python:3.10-slim

WORKDIR /app

# 编译部分依赖（asyncpg 等）可能需要的工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用 Docker 层缓存（代码变动时不必重装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷源码（.dockerignore 已排除 finetune/、data/、.venv 等）
COPY . .

EXPOSE 8000

# 默认启动 FastAPI 后端
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
