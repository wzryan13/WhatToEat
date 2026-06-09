"""
饮食管家 Agent — Streamlit Web 前端
"""
from __future__ import annotations

import asyncio
import logging
import re
import sys
from typing import Any

import streamlit as st

# ── 防呆：必须用 `streamlit run app.py` 启动，不能直接 `python app.py` ──
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        print(
            "\n❌ 启动方式错误。\n"
            "   请用：streamlit run app.py\n"
            "   不要用：python app.py\n",
            file=sys.stderr,
        )
        sys.exit(1)
except ImportError:
    pass
from langgraph.types import Command

from config.settings import settings
from graph import build_graph
from memory.store import get_memory_store, init_memory_store
from models.memory import ProfileUpdate
from rag.rag_service import init_rag_service
from tools import init_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ────────────────────────────────────────────────────────────────────────────
# 常量
# ────────────────────────────────────────────────────────────────────────────

ALL_NODES: list[str] = [
    "memory_read",
    "intent_parser",
    "search_agent",
    "rag_agent",
    "clarify",
    "error_output",
    "result_formatter",
    "rag_formatter",
    "memory_write",
]

SPICE_OPTIONS = ["不辣", "微辣", "中辣", "麻辣"]
SWEETNESS_OPTIONS = ["不甜", "微甜", "适中甜", "偏甜"]

CATEGORY_EMOJI = {
    "早餐": "☀️",
    "素菜": "🥗",
    "荤菜": "🍖",
    "主食": "🍚",
    "汤类": "🍲",
    "甜品": "🍰",
    "饮品": "🥤",
    "水产": "🐟",
    "半成品": "🥫",
    "调料": "🧂",
    "下午茶": "🧁",
}


def parse_comma_list(s: str) -> list[str]:
    """把 '花生, 海鲜，牛奶' 这种逗号分隔字符串切成 list，兼容中英文逗号。"""
    if not s:
        return []
    parts = [p.strip() for p in s.replace("，", ",").split(",")]
    return [p for p in parts if p]

# ────────────────────────────────────────────────────────────────────────────
# 页面配置 + 全局 CSS
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="饮食管家 Agent",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.chip {
    display: inline-block;
    padding: 3px 10px;
    margin: 2px 4px 2px 0;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
}
.chip-blue   { background: #1e3a8a40; color: #93c5fd; border: 1px solid #1e3a8a; }
.chip-red    { background: #7f1d1d40; color: #fca5a5; border: 1px solid #7f1d1d; }
.chip-yellow { background: #78350f40; color: #fcd34d; border: 1px solid #78350f; }
.chip-gray   { background: #37415140; color: #d1d5db; border: 1px solid #374151; }

.route-block {
    border-left: 3px solid #7c8cfa;
    background: #1f2329;
    padding: 10px 14px;
    margin: 8px 0 12px 0;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12.5px;
    line-height: 1.7;
    color: #cbd5e1;
}
.route-block .k { color: #7c8cfa; }
.route-block .v-green { color: #6ee7b7; }
.route-block .v-blue  { color: #93c5fd; }
.route-block .v-arrow { color: #94a3b8; }

.node-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 8px;
    margin: 2px 0;
    border-radius: 4px;
    background: #1f2329;
    font-family: ui-monospace, monospace;
    font-size: 12px;
}
.node-name { color: #cbd5e1; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 8px; }
.dot-done    { background: #22c55e; box-shadow: 0 0 6px #22c55e80; }
.dot-running { background: #3b82f6; box-shadow: 0 0 8px #3b82f6c0; animation: pulse 1.2s infinite; }
.dot-pending { background: transparent; border: 1px solid #4b5563; }
.node-status-done    { color: #22c55e; }
.node-status-running { color: #3b82f6; }
.node-status-pending { color: #6b7280; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.45; } }

.section-title {
    font-size: 12px;
    color: #94a3b8;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 16px;
    margin-bottom: 6px;
    text-transform: uppercase;
}
.brand-title { font-size: 22px; font-weight: 700; color: #e4e6eb; margin: 0; }
.brand-sub   { font-size: 11px; color: #94a3b8; margin: 0 0 4px 0; }

.restaurant-card-title { font-size: 16px; font-weight: 600; color: #e4e6eb; }
.restaurant-meta       { color: #94a3b8; font-size: 13px; margin-top: 2px; }
.restaurant-addr       { color: #cbd5e1; font-size: 12.5px; margin-top: 4px; }
.rating { color: #fbbf24; font-weight: 600; }

.recipe-title {
    font-size: 17px;
    font-weight: 700;
    color: #f3f4f6;
    letter-spacing: 0.3px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# 持久事件循环 + Agent 初始化（缓存一次）
# ────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="正在启动饮食管家 Agent...")
def get_runtime() -> dict[str, Any]:
    """初始化所有后端依赖，绑定到一个持久事件循环。"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _bootstrap():
        await init_tools()
        await init_memory_store()
        init_rag_service()  # 同步函数
        app = build_graph()
        store = get_memory_store()
        user_id = await store.get_or_create_user(
            settings.DEMO_CHANNEL, settings.DEMO_EXTERNAL_ID
        )
        runtime = await store.get_or_create_session(user_id)
        return app, store, user_id, runtime

    app, store, user_id, runtime = loop.run_until_complete(_bootstrap())
    return {
        "loop": loop,
        "app": app,
        "store": store,
        "user_id": user_id,
        "session_id": runtime.session_id,
        "thread_id": runtime.thread_id,
        "config": {"configurable": {"thread_id": runtime.thread_id}},
    }


def run_async(coro):
    """在持久事件循环上跑协程。"""
    return get_runtime()["loop"].run_until_complete(coro)


# ────────────────────────────────────────────────────────────────────────────
# 节点状态管理
# ────────────────────────────────────────────────────────────────────────────


def reset_node_status() -> None:
    st.session_state.node_status = {name: "pending" for name in ALL_NODES}
    st.session_state.node_order = []


def render_route_sidebar(container) -> None:
    """渲染侧栏的实时路由状态。"""
    status = st.session_state.get("node_status", {name: "pending" for name in ALL_NODES})
    html_parts = []
    for name in ALL_NODES:
        s = status.get(name, "pending")
        dot_cls = f"dot dot-{s}"
        mark = {"done": "✓", "running": "↻", "pending": "—"}[s]
        status_cls = f"node-status-{s}"
        html_parts.append(
            f'<div class="node-row">'
            f'<span><span class="{dot_cls}"></span><span class="node-name">{name}</span></span>'
            f'<span class="{status_cls}">{mark}</span>'
            f'</div>'
        )
    container.markdown("".join(html_parts), unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# 调用 Agent —— 流式订阅节点事件
# ────────────────────────────────────────────────────────────────────────────


async def _astream_invoke(payload, config, sidebar_container):
    """调用 graph 并实时刷新侧栏节点状态。返回 (final_state, interrupt_value)。"""
    app = get_runtime()["app"]
    node_set = set(ALL_NODES)

    async for event in app.astream_events(payload, config, version="v2"):
        kind = event.get("event", "")
        name = event.get("name", "")
        if name not in node_set:
            continue
        if kind == "on_chain_start":
            st.session_state.node_status[name] = "running"
            if name not in st.session_state.node_order:
                st.session_state.node_order.append(name)
            render_route_sidebar(sidebar_container)
        elif kind == "on_chain_end":
            st.session_state.node_status[name] = "done"
            render_route_sidebar(sidebar_container)

    snapshot = app.get_state(config)
    state = snapshot.values or {}

    interrupt_value = None
    for task in snapshot.tasks:
        if getattr(task, "interrupts", None):
            interrupt_value = task.interrupts[0].value
            break

    return state, interrupt_value


def invoke_agent(user_input: str, sidebar_container) -> tuple[dict, str | None]:
    """同步入口：根据当前 thread 状态选择 payload，返回 (final_state, interrupt_value)。"""
    rt = get_runtime()
    app, config = rt["app"], rt["config"]

    snapshot = app.get_state(config)
    has_checkpoint = bool(snapshot and getattr(snapshot, "values", None))
    is_interrupted = bool(snapshot.next) and any(
        n in snapshot.next for n in ("clarify",)
    )

    turn_no = run_async(rt["store"].next_turn(rt["session_id"]))

    if is_interrupted:
        payload = Command(resume=user_input)
    elif has_checkpoint:
        payload = {"user_input": user_input, "turn_no": turn_no}
    else:
        payload = {
            "user_id": rt["user_id"],
            "session_id": rt["session_id"],
            "thread_id": rt["thread_id"],
            "turn_no": turn_no,
            "user_input": user_input,
            "conversation_history": [],
        }

    reset_node_status()
    render_route_sidebar(sidebar_container)
    return run_async(_astream_invoke(payload, config, sidebar_container))


# ────────────────────────────────────────────────────────────────────────────
# 偏好编辑 —— 写回 UserProfile
# ────────────────────────────────────────────────────────────────────────────


def sync_preferences(
    spice: str | None,
    sweetness: str | None,
    allergies: list[str],
    blacklist: list[str],
    default_city: str | None,
) -> int:
    """计算 diff 并提交 ProfileUpdate 列表。"""
    rt = get_runtime()
    profile = run_async(rt["store"].load_profile(rt["user_id"]))

    updates: list[ProfileUpdate] = []

    cur_allergies = {f.value for f in profile.allergies}
    for v in set(allergies) - cur_allergies:
        updates.append(ProfileUpdate(field="allergies", action="add", value=v))
    for v in cur_allergies - set(allergies):
        updates.append(ProfileUpdate(field="allergies", action="remove", value=v))

    cur_black = {f.value for f in profile.food_blacklist}
    for v in set(blacklist) - cur_black:
        updates.append(ProfileUpdate(field="food_blacklist", action="add", value=v))
    for v in cur_black - set(blacklist):
        updates.append(ProfileUpdate(field="food_blacklist", action="remove", value=v))

    cur_spice = profile.spice_tolerance.value if profile.spice_tolerance else None
    if spice and spice != cur_spice:
        updates.append(ProfileUpdate(field="spice_tolerance", action="set", value=spice))
    elif not spice and cur_spice:
        updates.append(ProfileUpdate(field="spice_tolerance", action="remove", value=""))

    cur_sweet = profile.sweetness.value if profile.sweetness else None
    if sweetness and sweetness != cur_sweet:
        updates.append(ProfileUpdate(field="sweetness", action="set", value=sweetness))
    elif not sweetness and cur_sweet:
        updates.append(ProfileUpdate(field="sweetness", action="remove", value=""))

    cur_city = profile.default_city.value if profile.default_city else None
    if default_city and default_city.strip() and default_city.strip() != cur_city:
        updates.append(ProfileUpdate(field="default_city", action="set", value=default_city.strip()))

    if updates:
        run_async(rt["store"].apply_profile_updates(rt["user_id"], updates))
    return len(updates)


# ────────────────────────────────────────────────────────────────────────────
# UI 渲染：侧栏
# ────────────────────────────────────────────────────────────────────────────


def render_sidebar() -> "st.delta_generator.DeltaGenerator":
    """渲染整个侧栏，并返回「Agent 路由」区域的 placeholder（用于流式更新）。"""
    rt = get_runtime()
    profile = run_async(rt["store"].load_profile(rt["user_id"]))
    route_placeholder = None

    with st.sidebar:
        st.markdown(
            '<p class="brand-title">🍳 饮食管家</p>'
            '<p class="brand-sub">Dietary Butler Agent</p>',
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-title">👤 用户偏好</p>', unsafe_allow_html=True)

        cur_spice = profile.spice_tolerance.value if profile.spice_tolerance else None
        cur_sweet = profile.sweetness.value if profile.sweetness else None
        cur_allergies = [f.value for f in profile.allergies]
        cur_black = [f.value for f in profile.food_blacklist]
        cur_city = profile.default_city.value if profile.default_city else ""

        spice = st.selectbox(
            "辣度偏好",
            [""] + SPICE_OPTIONS,
            index=([""] + SPICE_OPTIONS).index(cur_spice) if cur_spice in SPICE_OPTIONS else 0,
        )
        sweetness = st.selectbox(
            "甜度偏好",
            [""] + SWEETNESS_OPTIONS,
            index=([""] + SWEETNESS_OPTIONS).index(cur_sweet) if cur_sweet in SWEETNESS_OPTIONS else 0,
        )
        allergies_text = st.text_input(
            "过敏（多个用逗号分隔）",
            value="，".join(cur_allergies),
            placeholder="如：花生，海鲜，牛奶",
        )
        allergies = parse_comma_list(allergies_text)

        blacklist_text = st.text_input(
            "忌口（多个用逗号分隔）",
            value="，".join(cur_black),
            placeholder="如：香菜，辣椒",
        )
        blacklist = parse_comma_list(blacklist_text)

        default_city = st.text_input("默认城市", value=cur_city, placeholder="如：成都")

        # 实时显示当前偏好的彩色 chip 预览
        chip_html = []
        for v in allergies:
            chip_html.append(f'<span class="chip chip-red">{v}</span>')
        for v in blacklist:
            chip_html.append(f'<span class="chip chip-yellow">{v}</span>')
        if spice:
            chip_html.append(f'<span class="chip chip-blue">{spice}</span>')
        if sweetness:
            chip_html.append(f'<span class="chip chip-blue">{sweetness}</span>')
        if chip_html:
            st.markdown(" ".join(chip_html), unsafe_allow_html=True)

        if st.button("💾 保存偏好", use_container_width=True):
            n = sync_preferences(
                spice or None, sweetness or None, allergies, blacklist, default_city
            )
            st.toast(f"已保存（{n} 项变更）" if n else "无变更", icon="✅")
            st.rerun()

        st.markdown('<p class="section-title">🔀 Agent 路由</p>', unsafe_allow_html=True)
        route_placeholder = st.empty()
        render_route_sidebar(route_placeholder)

        st.markdown('<p class="section-title">⚙ 技术栈</p>', unsafe_allow_html=True)
        st.markdown(
            '<span class="chip chip-gray">LangGraph</span>'
            '<span class="chip chip-gray">Milvus</span>'
            '<span class="chip chip-gray">PostgreSQL</span>'
            '<span class="chip chip-gray">高德 MCP</span>'
            '<span class="chip chip-gray">DeepSeek</span>'
            '<span class="chip chip-gray">Streamlit</span>',
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-title">会话</p>', unsafe_allow_html=True)
        st.caption(f"Session: `{rt['session_id'][-8:]}`")
        st.caption(f"Thread: `{rt['thread_id'][-8:]}`")

        if st.button("🗑 清空对话", use_container_width=True):
            st.session_state.messages = []
            reset_node_status()
            st.rerun()

    return route_placeholder


# ────────────────────────────────────────────────────────────────────────────
# UI 渲染：主区
# ────────────────────────────────────────────────────────────────────────────


def render_route_block(intent: str, loc_text: str, loc_type: str, route_nodes: list[str]) -> str:
    """返回紫色路由追踪块的 HTML。"""
    loc_line = ""
    if loc_text or loc_type:
        arrow = ' <span class="v-arrow">→</span> '
        loc_line = (
            f'<div><span class="k">location:</span> {loc_text or "—"}'
            f'{arrow}<span class="v-blue">{loc_type or "—"}</span></div>'
        )
    route_str = " <span class='v-arrow'>→</span> ".join(
        f'<span class="v-green">{n}</span>' for n in route_nodes
    )
    return (
        '<div class="route-block">'
        f'<div><span class="k">intent:</span>   <span class="v-blue">{intent or "—"}</span></div>'
        f'{loc_line}'
        f'<div><span class="k">route:</span>    {route_str}</div>'
        '</div>'
    )


def _price_tier(cost: str | None) -> str:
    if not cost:
        return ""
    try:
        c = float(cost)
    except (ValueError, TypeError):
        return ""
    if c < 50: return "$"
    if c < 100: return "$$"
    if c < 300: return "$$$"
    return "$$$$"


def render_restaurant_card(rec: dict, poi: dict) -> None:
    name = rec.get("name") or poi.get("name", "")
    rating = poi.get("rating", "")
    address = poi.get("address", "")
    cost = poi.get("cost", "")
    tier = _price_tier(cost)
    category = rec.get("category", "")
    reason = rec.get("reason", "")
    open_time = poi.get("opentime2") or poi.get("open_time", "")
    is_open = rec.get("is_open")
    distance = poi.get("distance", "")

    meta_parts = []
    if category: meta_parts.append(category)
    if tier: meta_parts.append(tier)
    if cost: meta_parts.append(f"人均 ¥{cost}")
    if distance: meta_parts.append(f"{distance}m")
    meta = " · ".join(meta_parts)

    rating_html = f'<span class="rating">⭐ {rating}</span>' if rating else ""
    open_badge = ""
    if is_open is True:
        open_badge = '<span class="chip chip-blue">营业中</span>'
    elif is_open is False:
        open_badge = '<span class="chip chip-yellow">可能未营业</span>'

    with st.container(border=True):
        col_l, col_r = st.columns([5, 1])
        with col_l:
            st.markdown(
                f'<div class="restaurant-card-title">{name} {open_badge}</div>'
                f'<div class="restaurant-meta">{meta}</div>'
                f'<div class="restaurant-addr">📍 {address}</div>',
                unsafe_allow_html=True,
            )
        with col_r:
            if rating_html:
                st.markdown(rating_html, unsafe_allow_html=True)
        if reason:
            st.markdown(f"💬 {reason}")
        if open_time:
            with st.expander("营业时间"):
                st.text(open_time)


_RE_STAR_DIFFICULTY = re.compile(r"预估烹饪难度[：:]\s*([★☆]+)")
_RE_CALORIES = re.compile(r"预估卡路里[：:]\s*([\d.]+)\s*大?卡")
_DIFFICULTY_TO_STARS = {"入门": 1, "简单": 2, "中等": 3, "较难": 4, "困难": 5}


def parse_recipe_markdown(content: str, difficulty_label: str = "") -> dict:
    """从菜谱 markdown 提取结构化字段。"""
    intro = ""
    stars = 0
    calories = ""
    ingredients: list[str] = []
    step_count = 0

    if not content:
        return {
            "intro": "", "stars": _DIFFICULTY_TO_STARS.get(difficulty_label, 0),
            "calories": "", "ingredients": [], "step_count": 0,
        }

    lines = content.splitlines()

    # 简介：第一个 # 标题之后的第一个非空、非标题、非 key:value、非列表项的段落
    _list_prefix = re.compile(r"^(-|\*|\d+\.)\s")
    found_title = False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            found_title = True
            continue
        if not found_title:
            continue
        if s.startswith(("预估烹饪难度", "预估卡路里")):
            continue
        if _list_prefix.match(s):
            # 遇到列表项还没找到简介，就当作没简介
            break
        intro = s
        break

    m = _RE_STAR_DIFFICULTY.search(content)
    if m:
        stars = m.group(1).count("★")
    elif difficulty_label in _DIFFICULTY_TO_STARS:
        stars = _DIFFICULTY_TO_STARS[difficulty_label]

    m = _RE_CALORIES.search(content)
    if m:
        calories = m.group(1).rstrip(".") + " 大卡"

    # 食材：取 ## 必备原料和工具 后的 bullet 列表
    in_ingredients_section = False
    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            in_ingredients_section = "必备原料" in s or "食材" in s
            continue
        if in_ingredients_section and (s.startswith("- ") or s.startswith("* ")):
            raw = s[2:].strip()
            # 去掉括号注释（如 "（可选）"）
            raw = re.sub(r"[（(].*?[)）]", "", raw).strip()
            # 取第一个空格 / 数字 / 标点 之前的部分作为食材名
            m = re.match(r"^([^\s,，0-9.。、:：/]+)", raw)
            if not m:
                continue
            name = m.group(1).strip(" ，,。.、")
            # 过短或过长都视为描述句，跳过
            if 1 < len(name) <= 6 and name not in ingredients:
                ingredients.append(name)

    # 操作步骤数
    in_steps_section = False
    step_re = re.compile(r"^\s*\d+\.\s")
    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            in_steps_section = "操作" in s or "步骤" in s
            continue
        if in_steps_section and step_re.match(line):
            step_count += 1

    return {
        "intro": intro,
        "stars": stars,
        "calories": calories,
        "ingredients": ingredients,
        "step_count": step_count,
    }


def render_stars(n: int, total: int = 5) -> str:
    """返回 ★ 实心 + ☆ 空心 的 HTML。"""
    n = max(0, min(total, n))
    return (
        '<span style="color:#fbbf24;">' + "★" * n + "</span>"
        '<span style="color:#4b5563;">' + "☆" * (total - n) + "</span>"
    )


def render_recipe_card(rec: dict) -> None:
    dish_name = rec.get("dish_name", "")
    category = rec.get("category", "")
    difficulty = rec.get("difficulty", "")
    reason = rec.get("reason", "")
    content = rec.get("content", "")

    parsed = parse_recipe_markdown(content, difficulty)
    emoji = CATEGORY_EMOJI.get(category, "🍽️")

    # 顶部：emoji + 菜名 + 分类 chip
    cat_chip = f'<span class="chip chip-blue">{category}</span>' if category else ""

    # 元数据行：星级 + 卡路里
    meta_parts = []
    if parsed["stars"]:
        meta_parts.append(
            f'{render_stars(parsed["stars"])} '
            f'<span style="color:#94a3b8;font-size:12px;">{difficulty}</span>'
        )
    if parsed["calories"]:
        meta_parts.append(
            f'<span style="color:#fb923c;">🔥</span> '
            f'<span style="color:#cbd5e1;font-size:12.5px;">{parsed["calories"]}</span>'
        )
    meta_line = '<span style="margin:0 12px;color:#374151;">|</span>'.join(meta_parts)

    # 食材标签云
    ingredients = parsed["ingredients"][:5]
    extra_count = len(parsed["ingredients"]) - len(ingredients)
    ing_chips = "".join(
        f'<span class="chip chip-gray">{ing}</span>' for ing in ingredients
    )
    if extra_count > 0:
        ing_chips += f'<span class="chip chip-gray">+{extra_count}</span>'

    with st.container(border=True):
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div class="recipe-title">{emoji}&nbsp;&nbsp;{dish_name}</div>'
            f'<div>{cat_chip}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if meta_line:
            st.markdown(
                f'<div style="margin:6px 0 4px 0;">{meta_line}</div>',
                unsafe_allow_html=True,
            )
        if parsed["intro"]:
            st.markdown(
                f'<div style="color:#9ca3af;font-size:13px;margin:6px 0 8px 0;'
                f'font-style:italic;">{parsed["intro"]}</div>',
                unsafe_allow_html=True,
            )
        if reason:
            st.markdown(
                f'<div style="color:#cbd5e1;font-size:13px;margin:4px 0 8px 0;">'
                f'💬 {reason}</div>',
                unsafe_allow_html=True,
            )
        if ingredients:
            st.markdown(
                f'<div style="color:#94a3b8;font-size:11px;font-weight:600;'
                f'margin-top:6px;">🛒 主要食材</div>'
                f'<div style="margin:4px 0;">{ing_chips}</div>',
                unsafe_allow_html=True,
            )
        step_label = (
            f"查看完整做法（共 {parsed['step_count']} 步）"
            if parsed["step_count"]
            else "查看完整菜谱"
        )
        if content:
            with st.expander(step_label):
                st.markdown(content)


def render_assistant_message(msg: dict) -> None:
    """渲染历史中的一条 assistant 消息（含路由块 + 卡片）。"""
    intent = msg.get("intent", "")
    loc_text = msg.get("location_text", "")
    loc_type = msg.get("location_type", "")
    route_nodes = msg.get("route", [])
    response_msg = msg.get("response_message", "")
    recs = msg.get("recommendations", [])
    pois = msg.get("pois", [])
    kind = msg.get("kind", "")

    st.markdown(
        render_route_block(intent, loc_text, loc_type, route_nodes),
        unsafe_allow_html=True,
    )

    if response_msg and kind != "restaurant":
        # 餐厅的 response_message 是冗长的 CLI 格式，不展示；菜谱/中断/错误正常展示
        # 但 restaurant 我们用自己的引导文字
        st.markdown(response_msg.split("\n")[0] if kind == "restaurant" else response_msg)
    elif kind == "restaurant" and recs:
        st.markdown(f"为你找到 {len(recs)} 家推荐：")

    if kind == "restaurant":
        poi_map = {p.get("id"): p for p in pois}
        for r in recs:
            render_restaurant_card(r, poi_map.get(r.get("id"), {}))
    elif kind == "recipe":
        # 两列网格
        for i in range(0, len(recs), 2):
            cols = st.columns(2, gap="small")
            with cols[0]:
                render_recipe_card(recs[i])
            if i + 1 < len(recs):
                with cols[1]:
                    render_recipe_card(recs[i + 1])


# ────────────────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────────────────


def build_assistant_msg(state: dict, interrupt_value: str | None) -> dict:
    """从最终 state 构造一条 assistant 消息记录。"""
    intent = state.get("intent_type", "")
    loc_text = state.get("location_text", "")
    loc_type = state.get("location_type", "")
    route = list(st.session_state.get("node_order", []))
    recs = state.get("final_recommendations", []) or []
    pois = state.get("filtered_pois") or state.get("detailed_pois") or []
    response_msg = state.get("response_message", "")

    if interrupt_value:
        kind = "interrupt"
        response_msg = interrupt_value
    elif intent in ("recipe", "recommend"):
        kind = "recipe"
    elif route and route[-1] == "memory_write" and "error_output" in route:
        kind = "error"
    elif recs:
        kind = "restaurant"
    else:
        kind = "empty"

    return {
        "role": "assistant",
        "intent": intent,
        "location_text": loc_text,
        "location_type": loc_type,
        "route": route,
        "response_message": response_msg,
        "recommendations": recs,
        "pois": pois,
        "kind": kind,
    }


def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "node_status" not in st.session_state:
        reset_node_status()
    if "awaiting_clarify" not in st.session_state:
        st.session_state.awaiting_clarify = False

    # 触发 runtime 初始化
    get_runtime()

    # 渲染侧栏并取回路由 placeholder（流式事件回调写入它）
    sidebar_route_placeholder = render_sidebar()

    # 主区标题
    st.markdown(
        '<h2 style="margin-bottom:0">🍱 饮食管家 Agent</h2>'
        '<p style="color:#94a3b8; margin-top:4px;">'
        '基于 LangGraph + RAG + 长期记忆的对话式饮食推荐系统</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # 历史消息
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                render_assistant_message(msg)

    # 输入框
    placeholder = (
        "请告诉我您所在的位置..."
        if st.session_state.awaiting_clarify
        else "试试：番茄炒蛋怎么做？"
    )
    user_input = st.chat_input(placeholder)

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Agent 思考中..."):
                state, interrupt_value = invoke_agent(user_input, sidebar_route_placeholder)

            msg = build_assistant_msg(state, interrupt_value)
            st.session_state.awaiting_clarify = bool(interrupt_value)
            st.session_state.messages.append(msg)
            render_assistant_message(msg)


main()
