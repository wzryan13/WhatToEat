import logging
from config.settings import settings
from get_MCP import AMapMCPToolProvider
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)

_tools: dict = {}
_provider: AMapMCPToolProvider | None = None


def get_tool_by_name(tools: list, name: str):
    for tool in tools:
        if tool.name == name:
            return tool
    return None


async def init_tools():
    """初始化 MCP 工具：建立长连接 session，所有 tool 调用复用同一个子进程。"""
    global _provider
    if _tools:
        return

    if not settings.AMAP_API_KEY:
        raise ValueError("AMAP_MAPS_API_KEY 未配置")

    _provider = AMapMCPToolProvider(api_key=settings.AMAP_API_KEY)
    session = await _provider.start()

    # 关键：传入 session 后，tool 工厂内部走 session.call_tool(...)，不再每次 spawn 新进程
    tools = await load_mcp_tools(session, server_name="amap_server")

    _tools["text_search"] = get_tool_by_name(tools, "maps_text_search")
    _tools["around_search"] = get_tool_by_name(tools, "maps_around_search")
    _tools["search_detail"] = get_tool_by_name(tools, "maps_search_detail")
    _tools["geo"] = get_tool_by_name(tools, "maps_geo")

    for name, tool in _tools.items():
        if tool is None:
            raise ValueError(f"工具 {name} 未找到，请检查工具名称")

    logger.info("[init_tools] 所有工具初始化完成（MCP 长连接复用）")


async def close_tools():
    """关闭 MCP 长连接，清理 npx 子进程。

    可选调用——进程退出时 OS 也会回收子进程，但主动调更优雅。
    """
    global _provider
    if _provider is not None:
        await _provider.close()
        _provider = None
    _tools.clear()
