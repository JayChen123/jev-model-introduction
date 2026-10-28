#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 JEV 一次决策的中间量全部打印出来
================================================================

运行：
    cd _05_jev
    uv run python trace_jev.py

配合 README「数学原理 → 实际例子」那一节看。它把下面这条链路上的每一步
都摊开打印，用的是 demo_primitives.py 里同一个工单分流问题：

    模板拼接 -> token 化 -> 主干前向 -> 取隐藏状态
        -> Wq/Wk 投影 -> 缩放点积 -> softmax -> 概率

照着打印出来的数自己按计算器验一遍，JEV 就没有黑箱了。
"""
import math

import torch

from jev_model import JevEngine, build_row, question_candidates, state_text

STATE = "客户反馈：订单被重复扣款两次，已经等了三天，要求立刻退款，否则就投诉。"
QUESTION = {
    "type": "choice",
    "instructions": "这个工单应该交给哪个团队处理？",
    "criteria": {
        "billing": "账单、发票、退款相关",
        "account": "账号登录、资料修改相关",
        "technical": "程序缺陷、接口集成相关",
        "other": "以上都不是",
    },
}


def render(state, question) -> str:
    """把 record 还原成送进模型的原始文本，方便肉眼核对模板。"""
    _, options = question_candidates(question)
    text = "State:\n" + state_text(state) + "\n"
    text += f"Question ({question['type']}):\n{question['instructions']}\n"
    for option in options:
        text += "Option:\n" + option + "\n"
    return text + "Decision:\n"


def rule(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main() -> None:
    engine = JevEngine.load()
    keys, options = question_candidates(QUESTION)

    # ---------------- 步骤 1 / 2：模板与 token ----------------
    rule("步骤 1：模板拼接（送进模型的就是这段纯文本）")
    print(render(STATE, QUESTION))

    row = build_row(engine.tokenizer, STATE, QUESTION)
    rule("步骤 2：token 化")
    print(f"序列长度 N = {len(row['ids'])}")
    print(f"p_dec（Decision: 末位）= {row['decision']}")
    for key, pos in zip(keys, row["option_ends"]):
        print(f"  p({key:<9}) = {pos}")
    print("序列最后 3 个 token:",
          [engine.tokenizer.decode([i]) for i in row["ids"][-3:]])

    # ---------------- 前向 ----------------
    ids = torch.tensor([row["ids"]], device=engine.device)
    positions = torch.tensor([row["positions"]], device=engine.device)
    with torch.inference_mode():
        hidden = engine.backbone(
            input_ids=ids,
            position_ids=positions,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        ).last_hidden_state[0]

    rule(f"步骤 3：取隐藏状态（每个 {hidden.shape[1]} 维）")
    head = hidden[row["decision"]]
    print(f"  h_dec     前4维 = {[round(v, 4) for v in head[:4].tolist()]}"
          f"   ‖h‖ = {head.float().norm():.2f}")
    for key, pos in zip(keys, row["option_ends"]):
        h = hidden[pos]
        print(f"  h[{key:<9}] 前4维 = {[round(v, 4) for v in h[:4].tolist()]}"
              f"   ‖h‖ = {h.float().norm():.2f}")
    print("  ↑ 5 个向量几乎一样：都是『读完一行 Option』的表示，差别只在那行文本")

    # ---------------- 步骤 4：投影 ----------------
    head_dtype = engine.head_query.weight.dtype
    with torch.inference_mode():
        query = engine.head_query(head.to(head_dtype))                      # u
        cand_keys = engine.head_key(hidden[row["option_ends"]].to(head_dtype))  # v_i

    rule(f"步骤 4：投影到 d = {engine.head_dim} 维决策空间")
    print(f"  u = Wq·h_dec + bq   前4维 = {[round(v, 4) for v in query[:4].float().tolist()]}"
          f"   ‖u‖ = {query.float().norm():.2f}")
    for key, vec in zip(keys, cand_keys):
        print(f"  v[{key:<9}]          前4维 = {[round(v, 4) for v in vec[:4].float().tolist()]}"
              f"   ‖v‖ = {vec.float().norm():.2f}")
    print("  ↑ 模长依旧接近，区别在**方向**上——所以用点积衡量匹配度")

    # ---------------- 步骤 5：缩放点积 ----------------
    # 与 jev_model.py 完全一致：点积和缩放都在半精度下算完，再转 float32 做 softmax
    dot = cand_keys @ query
    raw = dot.float()
    scaled = (dot / math.sqrt(engine.head_dim)).float()

    rule(f"步骤 5：缩放点积    sqrt(d) = sqrt({engine.head_dim}) = "
         f"{math.sqrt(engine.head_dim):.4f}")
    for key, raw_v, z in zip(keys, raw.tolist(), scaled.tolist()):
        print(f"  u·v[{key:<9}] = {raw_v:9.4f}    z = u·v/sqrt(d) = {z:8.4f}")
    print("  ↑ 原始点积跨度约 57，缩放后压到约 5，避免 softmax 直接饱和")

    # ---------------- 步骤 6：softmax ----------------
    exp_z = scaled.exp()
    total = exp_z.sum()
    probs = exp_z / total

    rule("步骤 6：softmax")
    for key, z, e, p in zip(keys, scaled.tolist(), exp_z.tolist(), probs.tolist()):
        print(f"  {key:<10} z={z:8.4f}   exp(z)={e:10.5f}   p={p:.6f}")
    print(f"  {'合计':<10} {'':>13}   Σexp={total:10.5f}   Σp={probs.sum():.6f}")

    # ---------------- 步骤 7：读结果 ----------------
    rule("步骤 7：读结果（choice → argmax，confidence → max p）")
    best = int(probs.argmax())
    print(f"  choice     = {keys[best]}")
    print(f"  confidence = {probs[best]:.6f}")
    print("  与 engine.predict 的返回对比：")
    print("   ", engine.predict({"state": STATE, "questions": {"route": QUESTION}})["route"])


if __name__ == "__main__":
    main()
