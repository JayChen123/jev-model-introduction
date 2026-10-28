#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JEV 决策模型本地推理 —— 教学用最小实现
================================================================

JEV 和聊天模型的根本区别：
    聊天模型：输入文本 -> 生成文本
    JEV    ：输入 state + 若干类型化问题 -> 输出每个问题的**概率分布**

它不"说话"，只做决策。所以调用方拿到的是可以直接写进 if/else 的数字，
不存在"模型输出格式不对、还要正则去解析"的问题。

三种接口任务（原语，可在一个请求里混用）：
    choice  从若干候选中选一个     -> probabilities / confidence / choice
    score   在有序量表上打分       -> probabilities / confidence / score / level
    noul    判断命题真假           -> probabilities / confidence / noul

模型怎么做到的（本文件就是完整实现，不依赖任何第三方 JEV 包）：
    1. 把 state、问题、每个候选项按固定模板拼成一段文本；
    2. 送进 text backbone，取出两个位置的隐藏状态：
         - 每个候选项描述结束处   -> 经 Wk 投影成候选"键" v_i
         - "Decision:" 结束处     -> 经 Wq 投影成决策"查询" u
    3. 做点积注意力并 softmax，得到候选集上的概率分布：
         z_i = dot(u, v_i) / sqrt(d)
         p   = softmax(z)

    关键点：候选项是通过**文本描述**参与计算的，不是固定的分类头权重。
    所以候选集可以在每次请求时临时给定、随时增删改——这就是 "dynamic candidates"。

输入格式（顶层 record）：
    {
        "state": "上下文，字符串 / dict / list 都可以",
        "questions": {
            "问题ID": {"type": "choice|score|noul", "instructions": "...", "criteria": ...},
            ...
        }
    }
    choice 的 criteria 是 {候选ID: 候选描述} 的字典（至少 2 个）
    score  的 criteria 是有序等级描述列表，从低到高（至少 2 档）
    noul   不需要 criteria，输出固定为 false/true 两档

本地 backbone：Qwen3.5-0.8B-JEV（约 0.8B，CPU / MPS / CUDA 都能跑）
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

# 本地轻量化 JEV 模型目录，可用环境变量 JEV_MODEL_DIR 覆盖
DEFAULT_MODEL_DIR = os.environ.get(
    "JEV_MODEL_DIR", "/Users/chan/projects/models/Qwen3.5-0.8B-JEV"
)

# 单行的最大 token 数（超过则直接报错，不静默截断——截断会把候选项切掉）
CUTOFF_LEN = 1024


# ----------------------------------------------------------------------
# 一、输入规整：把三种题型统一成「候选键 + 候选描述」
# ----------------------------------------------------------------------
def state_text(state) -> str:
    """state 支持字符串 / 对象 / 数组，后两者序列化成 JSON 文本。"""
    if isinstance(state, str):
        return state
    if isinstance(state, (dict, list)):
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    raise ValueError("state 必须是字符串、对象或数组")


def question_candidates(question: dict) -> tuple[list[str], list[str]]:
    """
    把问题统一成 (候选键列表, 候选描述列表)。

    三种题型在这里被抹平成同一种结构——对模型来说，
    它们都只是"一堆候选描述"加一个决策点，区别只在候选怎么来。
    """
    kind = question.get("type")
    criteria = question.get("criteria")

    if kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice 的 criteria 必须是至少含两个候选项的字典")
        keys = list(criteria)
        texts = [
            f"{key}: {value}" if value is not None else key
            for key, value in criteria.items()
        ]
    elif kind == "noul":
        # 真假判断：候选固定为 false / true 两档，描述可选
        keys = ["false", "true"]
        criteria = criteria or {}
        texts = [f"{key}: {criteria.get(key, key)}" for key in keys]
    elif kind == "score":
        # 量表打分：候选就是有序等级，键固定为 "0","1","2"...
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score 的 criteria 必须是至少两档的有序等级列表")
        keys = [str(i) for i in range(len(criteria))]
        texts = list(criteria)
    else:
        raise ValueError(f"不支持的题型：{kind}")

    return keys, texts


# ----------------------------------------------------------------------
# 二、把一条「state + 问题」编码成模型输入
# ----------------------------------------------------------------------
def build_row(tokenizer, state, question: dict) -> dict:
    """
    按 JEV 的固定模板拼出一行：

        State:
        <上下文>
        Question (<类型>):
        <问题说明>
        Option:
        <候选项1描述>
        Option:
        <候选项2描述>
        ...
        Decision:

    位置编码有个细节：state 占 0..L-1，每个问题的分支继续从 L 递增。
    因为每个问题单独成行（rows 执行模式），多个问题共享同一段 state 位置，
    彼此之间看不到对方的候选项，互不干扰。
    """
    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    state_ids = enc("State:\n" + state_text(state) + "\n")
    state_len = len(state_ids)

    ids = list(state_ids)
    ids += enc(f"Question ({question['type']}):\n{question['instructions']}\n")

    # 逐个候选项拼接，并记下每个候选项的**最后一个 token**位置
    _, options = question_candidates(question)
    option_ends = []
    for option in options:
        ids += enc("Option:\n" + option + "\n")
        option_ends.append(len(ids) - 1)

    # 决策后缀，记下它的最后一个 token 位置——查询向量就取自这里
    ids += enc("Decision:\n")
    decision_pos = len(ids) - 1

    # 位置 id：state 用 0..state_len-1，分支接着往下排
    positions = list(range(len(ids)))

    if len(ids) > CUTOFF_LEN:
        raise ValueError(
            f"输入需要 {len(ids)} token，超过上限 {CUTOFF_LEN}，请缩短 state 或候选描述"
        )
    return {
        "ids": ids,
        "positions": positions,
        "decision": decision_pos,
        "option_ends": option_ends,
    }


# ----------------------------------------------------------------------
# 三、推理引擎
# ----------------------------------------------------------------------
class JevEngine:
    """加载本地 Qwen3.5-0.8B-JEV，对 record 做一次前向得到概率分布。"""

    def __init__(self, backbone, tokenizer, head_query, head_key, head_dim, device, dtype):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.head_query = head_query
        self.head_key = head_key
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

    # ------------------------------ 加载模型 ------------------------------
    @classmethod
    def load(cls, model_dir: str | Path = DEFAULT_MODEL_DIR, *,
             device: str = "auto", dtype: str | None = None) -> "JevEngine":
        root = Path(model_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"找不到模型目录：{root}")

        # 设备自动选择：cuda > mps(Apple GPU) > cpu
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # 精度：CPU 上用 float32 最稳，GPU 上用半精度省显存
        if dtype is None:
            dtype = {"cuda": "bfloat16", "mps": "float16"}.get(device, "float32")
        torch_dtype = getattr(torch, dtype)

        manifest = json.loads((root / "decision_config.json").read_text())
        backbone = AutoModel.from_pretrained(
            root / "backbone",
            dtype=torch_dtype,
            attn_implementation=manifest.get("attn_implementation", "sdpa"),
        )
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 决策头：两个线性投影（query / key），参数量不随候选数量增长
        head_dim = int(manifest["head_dim"])
        hidden_size = backbone.config.hidden_size
        head_query = nn.Linear(hidden_size, head_dim)
        head_key = nn.Linear(hidden_size, head_dim)
        weights = load_file(root / "head.safetensors")
        head_query.load_state_dict(
            {"weight": weights["query.weight"], "bias": weights["query.bias"]}
        )
        head_key.load_state_dict(
            {"weight": weights["key.weight"], "bias": weights["key.bias"]}
        )

        for module in (backbone, head_query, head_key):
            module.to(device=device, dtype=torch_dtype).eval()

        return cls(backbone, tokenizer, head_query, head_key, head_dim, device, dtype)

    # ------------------------------ 单问题打分 ------------------------------
    @torch.inference_mode()
    def _score(self, state, question: dict) -> torch.Tensor:
        """对一个问题算出候选集上的概率分布，返回一维 tensor。"""
        row = build_row(self.tokenizer, state, question)
        ids = torch.tensor([row["ids"]], device=self.device)
        positions = torch.tensor([row["positions"]], device=self.device)

        hidden = self.backbone(
            input_ids=ids,
            position_ids=positions,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        ).last_hidden_state[0]

        head_dtype = self.head_query.weight.dtype
        # 查询向量：取自 "Decision:" 之后那个位置
        query = self.head_query(hidden[row["decision"]].to(head_dtype))
        # 候选键：取自每个候选项描述的结束位置
        keys = self.head_key(hidden[row["option_ends"]].to(head_dtype))

        # 缩放点积 + softmax —— 这就是整个决策逻辑
        logits = keys @ query / math.sqrt(self.head_dim)
        return logits.float().softmax(-1)

    # ------------------------------ 对外接口 ------------------------------
    def predict(self, record: dict, temperature: float = 1.0) -> dict:
        """
        对一条 record 里的所有问题做决策。

        temperature 只影响概率的"尖锐程度"：
            < 1.0 分布更集中；> 1.0 更平缓。
        注意它不改变候选之间的**排序**，只影响 confidence 的量级。
        """
        if "state" not in record:
            raise ValueError("record 必须包含 state")
        questions = record.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("record.questions 必须是非空字典")

        answers = {}
        for qid, question in questions.items():
            keys, _ = question_candidates(question)
            probs = (self._score(record["state"], question) / float(temperature))
            probs = probs.cpu().tolist()
            distribution = dict(zip(keys, probs))
            best = max(range(len(keys)), key=probs.__getitem__)

            answer = {
                "type": question["type"],
                "probabilities": distribution,
                # confidence = 最高概率，可用于"低置信度就转人工"
                "confidence": probs[best],
            }
            if question["type"] == "choice":
                answer["choice"] = keys[best]
            elif question["type"] == "noul":
                # noul 直接给出"命题为真"的概率，可当阈值用
                answer["noul"] = distribution["true"]
            else:  # score
                # 期望档位比 argmax 更平滑：可落在两档之间
                answer["score"] = sum(i * p for i, p in enumerate(probs))
                answer["level"] = best

            answers[qid] = answer
        return answers


if __name__ == "__main__":
    engine = JevEngine.load()
    record = {
        "state": "客户被重复扣款两次，要求立刻退款。",
        "questions": {
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
            "urgency": {
                "type": "score",
                "instructions": "这个工单的紧急程度如何？",
                "criteria": ["可以排队等待", "需要优先处理", "需要立即人工介入"],
            },
            "refund_requested": {
                "type": "noul",
                "instructions": "客户是否明确提出了退款请求？",
            },
        },
    }
    print(json.dumps(engine.predict(record), ensure_ascii=False, indent=2))
