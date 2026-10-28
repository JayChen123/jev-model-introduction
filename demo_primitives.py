#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JEV 三种接口任务的基础调用示例
================================================================

运行：
    cd _05_jev
    uv run python demo_primitives.py

本示例演示 JEV 最常用的三种题型（原语），并且**放在同一个请求里**：

    choice  从若干候选中选一个   —— 分类 / 路由 / 分流
    score   在有序量表上打分     —— 紧急度 / 满意度 / 风险分级
    noul    判断命题真假         —— 是否退款 / 是否需要人工介入

三个要点（JEV 和聊天模型最大的区别）：
    1. 一次请求共享同一份 state，多个问题各自独立判断，互不干扰；
    2. 候选项在**运行时**由你给定，随时增删改，不需要重新训练；
    3. 返回的是概率分布而不是文本，可以直接写进 if/else，不存在格式解析问题。
"""
import json

from jev_model import JevEngine

# ----------------------------------------------------------------------
# 打印辅助：把概率分布画成条形图，方便课堂上直观看结果
# ----------------------------------------------------------------------
def bar(p: float, width: int = 28) -> str:
    filled = int(round(p * width))
    return "█" * filled + "·" * (width - filled)


def pad(text: str, width: int) -> str:
    """中文按 2 个字符宽计算，让终端里的表格对齐。"""
    shown = sum(2 if ord(c) > 127 else 1 for c in text)
    return text + " " * max(0, width - shown)


def show(qid: str, question: dict, answer: dict) -> None:
    kind = answer["type"]
    print(f"\n[{kind}] {qid} —— {question['instructions']}")

    # 概率分布（按概率从高到低排列）
    for key, prob in sorted(answer["probabilities"].items(),
                            key=lambda kv: -kv[1]):
        print(f"    {pad(key, 14)}{prob:6.3f}  {bar(prob)}")

    # 各题型的主答案字段
    if kind == "choice":
        print(f"    -> 选择：{answer['choice']}")
    elif kind == "score":
        print(f"    -> 得分：{answer['score']:.2f}（最可能档位 {answer['level']}）")
    else:
        print(f"    -> P(为真) = {answer['noul']:.3f}")
    print(f"    -> confidence = {answer['confidence']:.3f}")


def main() -> None:
    print("加载本地 JEV 模型……")
    engine = JevEngine.load()          # 自动选设备：cuda > mps > cpu

    # ----------------------------------------------------------------
    # 一份 state + 三个类型化问题，一次调用全部拿到答案
    # ----------------------------------------------------------------
    record = {
        "state": "客户反馈：订单被重复扣款两次，已经等了三天，要求立刻退款，否则就投诉。",
        "questions": {
            # ① choice：分类 / 路由
            "route": {
                "type": "choice",
                "instructions": "这个工单应该交给哪个团队处理？",
                "criteria": {
                    "billing": "账单、发票、退款相关",
                    "account": "账号登录、资料修改相关",
                    "technical": "程序缺陷、接口集成相关",
                    "other": "以上都不是",
                },
            },
            # ② score：有序量表打分
            "urgency": {
                "type": "score",
                "instructions": "这个工单的紧急程度如何？",
                "criteria": [
                    "可以正常排队等待",
                    "需要优先处理",
                    "需要立即人工介入",
                ],
            },
            # ③ noul：真假判断
            "refund_requested": {
                "type": "noul",
                "instructions": "客户是否明确提出了退款请求？",
            },
        },
    }

    answers = engine.predict(record)

    print("\n" + "=" * 68)
    print("state:", record["state"])
    print("=" * 68)
    for qid, question in record["questions"].items():
        show(qid, question, answers[qid])

    # ----------------------------------------------------------------
    # 用法一：拿 choice 直接做分支路由
    # ----------------------------------------------------------------
    print("\n" + "-" * 68)
    print("用法 1：choice 做路由")
    team = answers["route"]["choice"]
    print(f"    工单转给 -> {team} 团队")

    # ----------------------------------------------------------------
    # 用法二：noul 的概率本身就是一个可直接比较的阈值
    # ----------------------------------------------------------------
    print("\n用法 2：noul 概率当阈值")
    if answers["refund_requested"]["noul"] > 0.8:
        print("    P(要求退款) > 0.8，自动创建退款工单")
    else:
        print("    P(要求退款) 不足，先让客服确认")

    # ----------------------------------------------------------------
    # 用法三：confidence 做三档路由——这是 JEV 在工程上最有价值的地方
    #   高 -> 自动执行；中 -> 人工复核；低 -> 升级给大模型 / 人工
    # 注意：confidence 只代表"分布有多集中"，不等于准确率，
    #       阈值必须用你自己的标注数据去标定。
    # ----------------------------------------------------------------
    print("\n用法 3：confidence 三档路由")
    conf = answers["route"]["confidence"]
    if conf >= 0.85:
        level, action = "高", "自动执行"
    elif conf >= 0.60:
        level, action = "中", "转人工复核"
    else:
        level, action = "低", "升级给人工 / 大模型处理"
    print(f"    confidence = {conf:.3f} -> 置信度{level} -> {action}")

    # ----------------------------------------------------------------
    # 用法四：动态候选——改候选项立刻生效，无需重新训练
    # ----------------------------------------------------------------
    print("\n" + "-" * 68)
    print("用法 4：动态候选（同一句话，换一套候选集，结果立刻变）")
    new_question = {
        "type": "choice",
        "instructions": "客户当前的情绪是什么？",
        "criteria": {
            "calm": "平静，只是询问",
            "dissatisfied": "不满，但还在沟通",
            "angry": "愤怒，已经威胁投诉或流失",
        },
    }
    new_answer = engine.predict(
        {"state": record["state"], "questions": {"emotion": new_question}}
    )["emotion"]
    show("emotion", new_question, new_answer)

    print("\n完整返回结构：")
    print(json.dumps(answers, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
