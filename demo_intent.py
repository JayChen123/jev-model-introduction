#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用本地 JEV 模型做意图识别（三分类）
================================================================

运行：
    cd _05_jev
    uv run python demo_intent.py

意图类别：
    general_technical          通用技术问题（编程、代码、算法、数据库、AI 等）
    professional_consultation  专业咨询（黑马教育的课程、学费、报名、校区等）
    other                      其他意图（闲聊、打招呼，以及上面两类都不是的）

为什么用 JEV 而不用聊天模型来做意图识别？
    聊天模型：让它"输出类别名"，再写正则去解析文本，还可能吐出"我认为这是……"之类
              的废话，或者编出一个不存在的类别。
    JEV    ：一次前向直接给出三个类别的概率分布，argmax 一定是合法类别，
              并且附带的 confidence 能让你对"没把握"的样本转人工。

JEV 官方对 choice 用法的两条建议，本例都遵守了：
    1. 问题必须是**原子的单维度判断**——一次只问一件事；
    2. 候选项要包含兜底项（这里的 other），避免被迫在都不合适的选项里选一个。
"""
from jev_model import JevEngine


def pad(text: str, width: int) -> str:
    """中文按 2 个字符宽计算，让终端里的表格对齐。"""
    shown = sum(2 if ord(c) > 127 else 1 for c in text)
    return text + " " * max(0, width - shown)


# ----------------------------------------------------------------------
# 三个意图类别的定义——这些描述文字会参与模型计算
# ----------------------------------------------------------------------
INTENT_LABELS = {
    "general_technical": "通用技术问题",
    "professional_consultation": "专业咨询",
    "other": "其他意图",
}

# choice 的 criteria：{稳定的类别 ID: 类别描述}
# 描述要写得具体、互斥，模型就是靠这些文字区分类别的
INTENT_CRITERIA = {
    "general_technical": (
        "编程与技术问题：编程语言用法、代码实现、代码报错排查、算法与数据结构、"
        "数据库、操作系统与计算机网络、人工智能与大模型技术。"
    ),
    "professional_consultation": (
        "黑马教育机构的业务咨询：课程内容、学费价格、优惠活动、报名方式与条件、"
        "开班时间、校区地址、师资情况、就业服务。"
    ),
    "other": (
        "以上都不是：日常闲聊、打招呼、天气、娱乐八卦、生活琐事，"
        "以及与编程技术和 IT 培训咨询都无关的其他话题。"
    ),
}

INSTRUCTIONS = """
判断用户这句话属于哪一类意图，必须且只能选择一个类别。

类别定义：
1. general_technical：用户在问编程、代码、算法、数据库、计算机网络、
   人工智能或大模型等技术问题。
2. professional_consultation：用户在咨询黑马教育或 IT 培训机构的课程、
   学费、优惠、报名、开班时间、校区地址、师资或就业服务。
3. other：上面两类都不是，例如闲聊、打招呼、天气、娱乐等。
"""


def build_request(query: str) -> dict:
    """把一句用户问题包装成一个 JEV 请求。"""
    return {
        "state": {"user_query": query.strip()},
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": INTENT_CRITERIA,
            }
        },
    }


def classify(engine: JevEngine, query: str, threshold: float = 0.60) -> dict:
    """
    返回一条用户问题的意图判断结果。

    threshold 是"自动放行"的置信度门槛，低于它就该转人工。
    这个值必须用你自己标注的数据去标定，不能拍脑袋定。
    """
    answer = engine.predict(build_request(query))["intent"]

    intent = answer["choice"]
    confidence = answer["confidence"]

    return {
        "query": query,
        # 三重保险：万一模型给出预期外的类别，兜底到 other
        "intent": intent if intent in INTENT_LABELS else "other",
        "intent_name": INTENT_LABELS.get(intent, INTENT_LABELS["other"]),
        "confidence": confidence,
        "probabilities": answer["probabilities"],
        # 置信度不够就交给人工，不要硬信模型
        "need_human": confidence < threshold,
    }


def main() -> None:
    print("加载本地 JEV 模型……")
    engine = JevEngine.load()

    queries = [
        # 通用技术问题
        "Python 怎么实现快速排序？",
        "帮我写一个 FastAPI 的 JWT 认证接口",
        "什么是 RAG，和传统搜索有什么区别？",
        "MySQL 索引失效一般是什么原因？",
        # 专业咨询
        "黑马的 AI 大模型课程多少钱？",
        "你们北京校区在哪里？",
        "现在报名有什么优惠吗？",
        "培训一般要学多久，包就业吗？",
        # 其他意图
        "你好，最近怎么样？",
        "今天天气怎么样？",
        "推荐几部好看的电影",
        "中午吃什么好呢",
    ]

    print("\n" + "=" * 72)
    print(pad("用户问题", 36) + pad("意图", 14) + pad("置信度", 10) + "来源")
    print("=" * 72)

    stats = {"auto": 0, "human": 0}
    for query in queries:
        result = classify(engine, query)
        source = "转人工" if result["need_human"] else "自动路由"
        stats["human" if result["need_human"] else "auto"] += 1
        print(pad(query, 36)
              + pad(result["intent_name"], 14)
              + pad(f"{result['confidence']:.3f}", 10)
              + source)

    print("=" * 72)
    print(f"自动路由 {stats['auto']} 条，转人工 {stats['human']} 条")

    # ----------------------------------------------------------------
    # 看一条具体结果的概率分布，理解 confidence 的含义
    # ----------------------------------------------------------------
    print("\n单条结果明细（置信度最高的那类占了多大比例）：")
    detail = classify(engine, "黑马的 Python 课程学费大概多少？")
    print(f"  问题：{detail['query']}")
    for key, prob in sorted(detail["probabilities"].items(), key=lambda kv: -kv[1]):
        filled = int(round(prob * 28))
        print(f"    {pad(INTENT_LABELS[key], 14)}{prob:6.3f}  "
              f"{'█' * filled}{'·' * (28 - filled)}")
    print(f"  -> 判定为「{detail['intent_name']}」，confidence = {detail['confidence']:.3f}")


if __name__ == "__main__":
    main()
