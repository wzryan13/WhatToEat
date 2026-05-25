import os
import logging
from contextlib import AsyncExitStack
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession

logger = logging.getLogger(__name__)


class AMapMCPToolProvider:
    """高德 MCP 工具提供者。

    持有长连接 stdio session，所有 tool 调用复用同一个 npx 子进程，
    避免每次 ainvoke 重新 spawn 进程（节省 400-1100ms / 次）。
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client: MultiServerMCPClient | None = None
        self.session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    async def start(self) -> ClientSession:
        """启动 MCP 子进程，建立长连接 session 并保持开启。

        Returns:
            已初始化的 ClientSession，调用方可传给 load_mcp_tools(session=...)
        """
        # 构建环境变量字典，将系统当前环境变量和高德 key 合并
        # 这样 npx 运行的子进程就能读取到该 key
        env = os.environ.copy()
        env["AMAP_MAPS_API_KEY"] = self.api_key

        self.client = MultiServerMCPClient({
            "amap_server": {
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y",                          # 自动确认安装
                    "@amap/amap-maps-mcp-server"   # 高德地图的包名
                ],
                "env": env,                        # 注入包含了 key 的环境变量
            }
        })

        logger.info("正在建立高德 MCP 长连接 session...")
        # AsyncExitStack：进入 session 上下文但不立即退出，长期持有
        # close() 时由 stack.aclose() 统一清理（关闭 stdio 管道 + 终止 npx 子进程）
        self._exit_stack = AsyncExitStack()
        self.session = await self._exit_stack.enter_async_context(
            self.client.session("amap_server")
        )
        logger.info("高德 MCP 长连接 session 已建立，后续 tool 调用复用此 session")
        return self.session

    async def close(self):
        """关闭长连接、清理 npx 子进程。"""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self.session = None
            self.client = None
            logger.info("高德 MCP 长连接 session 已关闭")
