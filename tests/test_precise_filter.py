"""
针对 nodes.search_agent._precise_filter 的单元测试。

目的：暴露价格/评分硬过滤里容易踩到的边界与脏数据问题。
测试既覆盖"应该通过/应该过滤"的常规逻辑，也专门标注了若干
"当前实现行为 vs 理想行为"分歧的用例（用 xfail 标记并写明原因），
方便 Ryan 决定哪些是真 bug、哪些是产品语义需要讨论的点。

运行：
    python3 -m pytest tests/test_precise_filter.py -v
"""
import pytest

from models.intent import FilterConditions
from nodes.search_agent import _precise_filter


def poi(cost=None, rating=None, name="测试店", poi_id="p1", poi_type="餐饮服务;中餐"):
    return {
        "id": poi_id,
        "name": name,
        "cost": cost,
        "rating": rating,
        "type": poi_type,
    }


# ───────────────────────── Group A · 基本逻辑正确性 ─────────────────────────


class TestBasicPrice:
    def test_empty_filters_keeps_all(self):
        pois = [poi(cost="50"), poi(cost="200")]
        assert len(_precise_filter(pois, {})) == 2

    def test_price_max_passes_when_under(self):
        assert _precise_filter([poi(cost="50")], {"price_max": 100}) == [poi(cost="50")]

    def test_price_max_filters_when_over(self):
        assert _precise_filter([poi(cost="150")], {"price_max": 100}) == []

    def test_price_min_filters_when_under(self):
        assert _precise_filter([poi(cost="30")], {"price_min": 50}) == []

    def test_price_min_passes_when_over(self):
        assert _precise_filter([poi(cost="80")], {"price_min": 50}) == [poi(cost="80")]

    def test_price_range_both(self):
        pois = [poi(cost="30", poi_id="a"), poi(cost="80", poi_id="b"), poi(cost="200", poi_id="c")]
        result = _precise_filter(pois, {"price_min": 50, "price_max": 100})
        assert [p["id"] for p in result] == ["b"]


class TestBasicRating:
    def test_rating_passes(self):
        assert _precise_filter([poi(rating="4.6")], {"min_rating": 4.5}) == [poi(rating="4.6")]

    def test_rating_filters(self):
        assert _precise_filter([poi(rating="4.0")], {"min_rating": 4.5}) == []

    def test_rating_and_price_combined(self):
        # cost 通过但 rating 不达标 → 应被过滤
        p = poi(cost="50", rating="3.0")
        assert _precise_filter([p], {"price_max": 100, "min_rating": 4.0}) == []


# ───────────────────── Group B · cost 字段脏数据（用户报错重灾区） ─────────────────────


class TestDirtyCost:
    """
    高德 POI 实际返回的 cost 字段非常不规范：可能为空字符串、区间字符串如 "50-100"、
    None、甚至非法文本。当前 _precise_filter 用 try/except 兜住 ValueError，但被
    兜住意味着「过滤静默失效，POI 通过」，这是用户描述"指定价格还会返回奇怪结果"
    的可能根因之一。
    """

    def test_cost_none_passes_due_to_falsy_short_circuit(self):
        # cost=None → `if cost` 为 False，整段跳过 → POI 通过
        # 这是当前实现的「语义可接受」行为：cost 未知 ≠ 一定超预算
        assert _precise_filter([poi(cost=None)], {"price_max": 100}) == [poi(cost=None)]

    def test_cost_empty_string_passes_due_to_falsy_short_circuit(self):
        # cost="" → `if cost` 为 False（空串 falsy）→ 同上通过
        assert _precise_filter([poi(cost="")], {"price_max": 100}) == [poi(cost="")]

    @pytest.mark.xfail(
        strict=True,
        reason="潜在 bug：cost='50-100' 区间字符串导致 float() 抛 ValueError，被 except 吞掉，POI 通过过滤；"
               "但产品语义上 price_max=80 时这家店区间上限超 80 应被过滤（或至少打 warning）",
    )
    def test_cost_range_string_should_be_filtered(self):
        # 高德返回 cost='50-100' 时，price_max=80 应过滤掉
        assert _precise_filter([poi(cost="50-100")], {"price_max": 80}) == []

    @pytest.mark.xfail(
        strict=True,
        reason="潜在 bug：cost='abc' 等非法字符串 float() 失败被吞，POI 仍通过；建议至少视为未知或丢弃",
    )
    def test_cost_invalid_string_should_be_filtered(self):
        assert _precise_filter([poi(cost="abc")], {"price_max": 100}) == []

    def test_cost_numeric_string_works(self):
        assert _precise_filter([poi(cost="80")], {"price_max": 100}) == [poi(cost="80")]

    def test_cost_int_works(self):
        # 当前实现 float(int) 没问题，应通过
        assert _precise_filter([poi(cost=80)], {"price_max": 100}) == [poi(cost=80)]

    def test_cost_float_works(self):
        assert _precise_filter([poi(cost=80.5)], {"price_max": 100}) == [poi(cost=80.5)]


# ───────────────────── Group C · price=0 边界（potential bug） ─────────────────────


class TestZeroBoundary:
    """
    边界回归：price_max=0 / price_min=0 时过滤逻辑不应被 falsy 短路。
    当前代码用 `price_max is not None` 已正确处理，这组测试用于回归守护，
    防止有人改回 `if filters.get("price_max") and ...` 的旧写法。
    """

    def test_price_max_zero_filters_all_positive_cost(self):
        # price_max=0 表示"免费"，cost=50 的店应被过滤
        assert _precise_filter([poi(cost="50")], {"price_max": 0}) == []

    def test_price_min_zero_keeps_all(self):
        # price_min=0 任何非负 cost 都通过
        assert _precise_filter([poi(cost="50")], {"price_min": 0}) == [poi(cost="50")]


# ───────────────────── Group D · rating 字段脏数据 ─────────────────────


class TestDirtyRating:
    def test_rating_none_passes(self):
        # rating=None → `if rating` 跳过 → 通过
        assert _precise_filter([poi(rating=None)], {"min_rating": 4.0}) == [poi(rating=None)]

    def test_rating_empty_string_passes(self):
        # rating="" → 空串 falsy 跳过
        assert _precise_filter([poi(rating="")], {"min_rating": 4.0}) == [poi(rating="")]

    @pytest.mark.xfail(
        strict=True,
        reason="潜在 bug：rating='暂无' 等非数字字符串 float() 失败被吞，POI 仍通过；建议视为未知",
    )
    def test_rating_invalid_string_should_be_handled(self):
        assert _precise_filter([poi(rating="暂无")], {"min_rating": 4.0}) == []

    def test_rating_boundary_equal_passes(self):
        # 评分等于阈值 → 当前用 < 比较 → 通过
        assert _precise_filter([poi(rating="4.0")], {"min_rating": 4.0}) == [poi(rating="4.0")]


# ───────────────────── Group E · FilterConditions 对象兼容性 ─────────────────────


class TestFilterConditionsObject:
    """search_agent.py 里 filters 来自 IntentParserOutput.filters（FilterConditions 实例），
    虽然代码里 `state.get("filters") or {}` 会兜底 dict，但正常路径下传的是 pydantic 对象。
    FilterConditions 自带 `.get()` 方法，要确保两种类型都能正确过滤。"""

    def test_with_pydantic_object_price(self):
        fc = FilterConditions(price_max=100)
        assert _precise_filter([poi(cost="50")], fc) == [poi(cost="50")]
        assert _precise_filter([poi(cost="200")], fc) == []

    def test_with_pydantic_object_rating(self):
        fc = FilterConditions(min_rating=4.5)
        assert _precise_filter([poi(rating="4.6")], fc) == [poi(rating="4.6")]
        assert _precise_filter([poi(rating="4.0")], fc) == []

    def test_with_pydantic_object_combined(self):
        fc = FilterConditions(price_max=100, min_rating=4.0)
        pois = [
            poi(cost="50", rating="4.5", poi_id="ok"),
            poi(cost="50", rating="3.0", poi_id="lowrating"),
            poi(cost="200", rating="4.5", poi_id="expensive"),
        ]
        result = _precise_filter(pois, fc)
        assert [p["id"] for p in result] == ["ok"]


# ───────────────────── Group F · 大批量与排序保持 ─────────────────────


class TestBatch:
    def test_order_preserved(self):
        pois = [poi(cost="50", poi_id=f"p{i}") for i in range(5)]
        result = _precise_filter(pois, {"price_max": 100})
        assert [p["id"] for p in result] == ["p0", "p1", "p2", "p3", "p4"]

    def test_all_filtered_returns_empty(self):
        pois = [poi(cost="500", poi_id=f"p{i}") for i in range(5)]
        assert _precise_filter(pois, {"price_max": 100}) == []

    def test_empty_input(self):
        assert _precise_filter([], {"price_max": 100}) == []
