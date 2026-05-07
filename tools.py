import logging
from config.settings import settings
from get_MCP import AMapMCPToolProvider  # 替换为实际包名

logger = logging.getLogger(__name__)

_tools: dict = {}


def get_tool_by_name(tools: list, name: str):
    for tool in tools:
        if tool.name == name:
            return tool
    return None


async def init_tools():
    if _tools:
        return

    if not settings.AMAP_API_KEY:
        raise ValueError("AMAP_MAPS_API_KEY 未配置")

    toolkit = AMapMCPToolProvider(api_key=settings.AMAP_API_KEY)
    tools = await toolkit.get_tools()

    _tools["text_search"] = get_tool_by_name(tools, "maps_text_search")
    _tools["around_search"] = get_tool_by_name(tools, "maps_around_search")
    _tools["search_detail"] = get_tool_by_name(tools, "maps_search_detail")
    _tools["geo"] = get_tool_by_name(tools, "maps_geo")

    for name, tool in _tools.items():
        if tool is None:
            raise ValueError(f"工具 {name} 未找到，请检查工具名称")

    logger.info("[init_tools] 所有工具初始化完成")