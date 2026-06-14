# datasets/raw — 原始 query 种子

把你手挑的原始用户 query 放这里（这个目录会进 git，作为数据资产）。

## 建议格式

`queries.txt`，一行一条原始口语 query，例如：

```
今晚吃啥
有鸡蛋和西红柿能做什么
想吃辣的川菜
教我做红烧肉
来点清淡的素菜
新手能做的简单菜
```

## 覆盖面（尽量每类都有）

对应主项目的 6 种 `query_type`：
`specific_dish`（具体菜名）/ `scene`（场景）/ `flavor`（口味）/
`ingredient`（食材）/ `category`（菜系大类）/ `difficulty`（难度）。

越「脏」越好：口语、省略、错别字、模糊——贴近真实用户输入，student 才学得鲁棒。

> 放好后告诉我，我来把它接进 `scripts/1_build_dataset.py` 并转成训练用的 jsonl。
