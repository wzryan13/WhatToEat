# 饮食管家 Agent 设计总结

---

## 一、项目结构

```
your_project/
├── config/
│   ├── __init__.py
│   ├── prompts.py          # 所有节点prompt模板
│   └── settings.py         # API keys、默认参数
│
├── memory/                 # 预留，暂不实现
│   └── user_profile.py
│
├── models/
│   ├── __init__.py
│   ├── intent.py           # IntentParserOutput、FilterConditions
│   ├── poi.py              # POIBasic、POIDetail
│   ├── rerank.py           # LLMRerankOutput、Recommendation
│   └── state.py            # DietState
│
├── nodes/
│   ├── __init__.py
│   ├── intent_parser.py
│   ├── search.py           # landmark_resolver、keyword_search、around_search
│   ├── filter.py           # batch_poi_detail、precise_filter、llm_rerank
│   └── output.py           # clarify、error_output、result_formatter
│
├── tools.py                # init_tools、_tools、get_tool_by_name
├── graph.py                # graph连接、路由函数、build_graph
└── main.py                 # 入口、对话loop
```

---

## 二、Graph 文字图

```
START
  ↓
[intent_parser] 意图解析
  输出: location_text, city(仅城市名),
        keywords(地标+品类词 或 纯品类词，多角度扩展3-6个),
        search_mode("keyword"|"around" 由LLM决定),
        filters(price_max, price_min, radius, open_time, min_rating),
        negative_conditions, intent_type,
        location_type("valid"|"relative"|"gps"|"none"|"invalid"),
        has_contradiction, contradiction_message,
        current_time(系统注入)
  ↓
[route_after_intent] 路由
  ├─ invalid ──────────────────────────→ [error_output] → END
  ├─ none/relative/gps
  │   ├─ 追问次数 >= MAX_CLARIFICATION → [error_output] → END
  │   └─ 追问次数 < MAX_CLARIFICATION  → [clarify]
  │                                          ↓ interrupt暂停，等待用户输入
  │                                          ↓ resume后回到intent_parser
  └─ valid
      ├─ search_mode="around" ──────────→ [landmark_resolver]
      └─ search_mode="keyword" ─────────→ [keyword_search]

[landmark_resolver] maps_geo获取经纬度
  ├─ 成功 ──────────────────────────────→ [around_search]
  │                                         并发多keyword周边搜，合并去重
  └─ 失败 → 降级软提示 ─────────────────→ [keyword_search]
                                            并发多keyword搜索，合并去重

[keyword_search] / [around_search] 结果汇聚
  ↓
[batch_poi_detail] 并发查全量POI详情（上限15条）
  ↓
[precise_filter] 硬过滤
  price_max / price_min / min_rating 字段硬过滤
  营业时间不硬解析，opentime2 + open_time 留给 llm_rerank
  ↓
[llm_rerank] LLM综合筛选
  传入: 用户原始输入 + negative_conditions
        + 每家POI完整信息(name/type/cost/rating/opentime2/open_time/address)
        + current_time
  处理: type软过滤 / 负向条件 / 场景氛围
        营业时间判断(opentime2优先，open_time兜底)
        用户画像(预留长记忆)
  输出: 排序后推荐列表(最多5家) + 每家推荐理由 + disclaimer
  ↓
[result_formatter] 结果格式化输出
  附加: 矛盾条件软提示 / 忌口免责提示 / 降级提示 / 结果不足提示
  ↓
END
```

---

## 三、核心设计决策

### 3.1 工具层

| 工具 | 用途 |
|------|------|
| maps_geo | landmark_resolver，地名 → 经纬度 |
| maps_text_search | keyword_search，并发多关键词搜索 |
| maps_around_search | around_search，周边搜索 |
| maps_search_detail | batch_poi_detail，获取POI完整信息 |

工具初始化采用异步函数 `init_tools()`，应用启动时调用一次，存入 `_tools` 字典，避免重复初始化。

### 3.2 搜索策略

- `search_mode` 由 `intent_parser` 里的LLM决定，不用规则判断
- `around` 仅当用户明确表达附近/周边/距离范围时触发
- 并发多关键词搜索后合并去重，结果池约15-30条
- 无论是否有过滤条件，全量调用POI详情（上限15条）
- `landmark_resolver` 解析失败时降级为关键词搜，附软提示

### 3.3 关键词生成策略（intent_parser）

| 用户输入类型 | 处理方式 |
|-------------|---------|
| 明确品类（烧烤） | 主品类 + 细分品类 + 氛围词扩展，如["烧烤","韩式烤肉","炭火烤串","大排档"] |
| 口味描述（清淡） | LLM转译为具体菜系，如["粤菜","淮扬菜","潮汕菜","日料","茶餐厅"] |
| 场景类（约会） | LLM转译为适合场景的品类词，如["西餐","日料","法餐","意大利菜"] |
| 时段类（夜宵） | LLM转译为该时段常见品类，如["烧烤","大排档","麻辣烫","夜宵"] |
| 无品类 | ["热门美食","特色小吃","地方特色菜"] |
| 价格导向（便宜） | LLM转译为平价品类词，如["苍蝇馆子","粉面馆","简餐","小吃"] |
| 品牌 | 直接用品牌名，不扩展，intent_type标记为brand |
| 负向条件（不辣） | 转为正向品类词，原条件存negative_conditions |
| 有具体地标 | 地标+品类词拼在一起，如"万象城烧烤"、"春熙路火锅" |
| 只有城市 | 纯品类词，city参数传城市名 |

**关键词生成硬规则：**

1. 每个keyword单独搜索要有意义，不能是描述词如"清淡美食"
2. 不能把多个描述拼在一起如"日韩烤肉"，要拆开为"韩式烤肉"、"日式烤肉"
3. keywords数量3-6个
4. 城市名不单独混入keyword，具体地标要拼入keyword
5. 负向条件不进keywords，单独存入negative_conditions

### 3.4 筛选机制

```
硬过滤（precise_filter）          软过滤（llm_rerank）
─────────────────────             ─────────────────────
price_max / price_min             type字段软过滤
min_rating                        负向条件（不辣/忌口）
                                  场景氛围（约会/包间）
                                  营业时间（opentime2+open_time）
                                  用户画像（预留长记忆）
```

### 3.5 位置处理

| location_type | 处理方式 |
|--------------|---------|
| valid | 直接进入搜索流程 |
| relative（公司楼下） | 查长记忆（预留）→ 无则追问 |
| gps（当前位置） | GPS定位（预留）→ 无则追问 |
| none | 追问，最多1次（MAX_CLARIFICATION=1） |
| invalid（月球等） | 直接报错返回 |

### 3.6 追问机制（clarify节点）

- 使用LangGraph `interrupt` 机制暂停图执行，不在图内部循环等待
- `interrupt(message)` 将追问话术返回给用户，图挂起，state持久化
- 用户回复后 resume，带着新输入重新进入 `intent_parser`
- 最多追问1次，超过次数直接走 `error_output`
- checkpointer使用 `MemorySaver`，生产环境可换 `SqliteSaver` / `PostgresSaver`
- 每个对话session通过 `thread_id` 标识，checkpointer据此恢复state

### 3.7 营业时间处理

- 不做硬解析，避免格式不统一导致的脆性
- `opentime2`（完整版含星期/节假日）和 `open_time`（简化版）都传给 `llm_rerank`
- 以 `opentime2` 为主，`open_time` 为空时降级使用
- `current_time` 在 `intent_parser` 节点注入，格式：`YYYY-MM-DD HH:MM 星期X`

---

## 四、State 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| user_input | str | 用户最新输入 |
| conversation_history | list[dict] | 多轮对话历史 |
| current_time | str | 系统注入当前时间 |
| intent_type | Literal | normal/brand/scene/time_based |
| location_text | str\|None | 原始位置文本 |
| location_type | Literal | valid/relative/gps/none/invalid |
| city | str\|None | 仅城市名 |
| keywords | list[str] | 高德搜索关键词，3-6个 |
| search_mode | Literal | keyword/around |
| filters | FilterConditions | 结构化过滤条件 |
| negative_conditions | list[str] | 负向条件 |
| has_contradiction | bool | 是否有矛盾条件 |
| contradiction_message | str\|None | 矛盾说明 |
| clarification_count | int | 已追问次数 |
| landmark_resolve_failed | bool | 地标解析是否失败 |
| landmark_location | str\|None | 经纬度字符串 |
| raw_pois | list[dict] | 搜索原始结果 |
| detailed_pois | list[dict] | POI详情结果 |
| filtered_pois | list[dict] | 硬过滤后结果 |
| final_recommendations | list[dict] | 最终推荐列表 |
| response_message | str | 回复给用户的话术 |
| disclaimer_needed | bool | 是否需要免责提示 |
| disclaimer_message | str\|None | 免责提示内容 |
| result_insufficient | bool | 结果不足标记 |
| error_message | str\|None | 错误信息 |

---

## 五、Pydantic 模型结构

```
models/
├── intent.py
│   ├── FilterConditions        # price_max/min, radius, open_time, min_rating
│   └── IntentParserOutput      # 意图解析节点输出，用于with_structured_output
│
├── poi.py
│   ├── POIBasic                # 关键词搜/周边搜基础字段
│   └── POIDetail               # POI详情完整字段
│
├── rerank.py
│   ├── Recommendation          # 单条推荐：id/name/reason/is_open
│   └── LLMRerankOutput         # rerank节点输出，用于with_structured_output
│
└── state.py
    └── DietState               # LangGraph state，total=False允许部分更新
```

所有LLM输出节点（`intent_parser`、`llm_rerank`）使用 `llm.with_structured_output(PydanticModel)` 替代手动 `json.loads`，保证类型安全和字段验证。

---

## 六、配置参数（settings.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MODEL_NAME | claude-opus-4-5 | LLM模型名称 |
| AMAP_API_KEY | - | 高德地图API Key |
| POI_DETAIL_LIMIT | 15 | 并发POI详情上限 |
| MAX_RECOMMENDATIONS | 5 | 最终推荐餐厅数量上限 |
| MAX_CLARIFICATION | 1 | 最大追问次数 |
| DEFAULT_RADIUS | 1000 | 周边搜默认半径（米） |

---

## 七、预留扩展点

| 模块 | 位置 | 说明 |
|------|------|------|
| 长记忆-位置 | info_complement节点 | 补全relative/gps类型的location |
| 长记忆-用户画像 | llm_rerank节点 | 饮食偏好/忌口/健康状态等个性化推荐 |
| GPS定位 | info_complement节点 | 获取用户真实经纬度 |
| RAG知识库 | intent_parser节点 | 高德类别表辅助关键词转译，丰富口味/场景映射 |
| 持久化checkpointer | graph.py | MemorySaver → SqliteSaver/PostgresSaver |
| 多轮refine | result_formatter后 | 用户"再推荐几家"等场景的interrupt机制 |
