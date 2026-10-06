"""文档自动分类器。

对应平台需求的设计要点：

1. **少样本定义类别**：用户为每个类别提供少量示例，分类器采用
   类别质心（Rocchio）模型 —— 类别向量 = 该类全部示例的 TF-IDF 质心，
   新文档按与各类别质心的余弦相似度归类。
2. **增量学习，无需全库重学**：只维护两类充分统计量 ——
   各类别的词项频次和（``tf``）与文档频次（``df``）。
   新增 / 删除示例都是 O(文档词数) 的原子更新；IDF 在分类时按当前
   统计量现算，保证「换一批示例，类别边界立即跟着调整」，
   且增量状态与全量重建**逐位一致**（词频为整数，无浮点累积误差）。
3. **确定性**：分词、打分、排序全程无随机性；打分按类别 id 字典序
   迭代，并列时按类别名与 id 决胜。同一文档在同一模型状态下
   多次分类，结果与置信度完全一致。
4. **可拒识 + 多候选**：最高相似度低于阈值时不硬塞类别
   （``accepted=False``）；头部类别置信度差距小于 ``margin`` 时
   返回多个候选并置 ``ambiguous=True``。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
from collections import Counter
from typing import Iterable, Optional

from .segmenter import Segmenter
from .text import filter_stopwords


class DocumentClassifier:
    """基于类别质心的增量式文档分类器。"""

    MODEL_VERSION = 1

    def __init__(self, segmenter: Optional[Segmenter] = None,
                 temperature: float = 0.1,
                 accept_threshold: float = 0.1,
                 margin: float = 0.15):
        self.segmenter = segmenter
        # softmax 温度：越小置信度越尖锐
        self.temperature = float(temperature)
        # 余弦相似度低于该值时拒识（不强行归类）
        self.accept_threshold = float(accept_threshold)
        # 头部类别置信度差距小于该值时判为边界模糊（多候选）
        self.margin = float(margin)
        # 类别 id -> {"name", "tf": 词项频次和, "df": 词项文档频次, "n_docs"}
        self.categories: dict[str, dict] = {}
        # 全部示例上的文档频次（各类别 df 之和）
        self.df: Counter = Counter()
        self.n_docs = 0
        self._lock = threading.RLock()

    # -- 文本预处理 ---------------------------------------------------------
    def tokenize(self, text: str) -> list[str]:
        """分词 + 去停用词 + 拉丁字母小写化（确定性）。"""
        if self.segmenter is None:
            self.segmenter = Segmenter()
        words = self.segmenter.cut(text or "")
        return [w.lower() for w in filter_stopwords(words)]

    # -- 类别管理 -----------------------------------------------------------
    def add_category(self, category_id: str, name: str = "") -> None:
        """登记类别；已存在时仅更新名称（幂等）。"""
        with self._lock:
            cat = self.categories.get(category_id)
            if cat is None:
                self.categories[category_id] = {
                    "name": name, "tf": Counter(), "df": Counter(), "n_docs": 0,
                }
            elif name:
                cat["name"] = name

    def remove_category(self, category_id: str) -> None:
        """移除类别及其全部统计量（幂等）。"""
        with self._lock:
            cat = self.categories.pop(category_id, None)
            if not cat:
                return
            for token, c in cat["df"].items():
                self._decr(self.df, token, c)
            self.n_docs -= cat["n_docs"]

    # -- 增量学习 -----------------------------------------------------------
    def add_example(self, category_id: str, text: str) -> int:
        """学习一条示例，返回有效词项数。"""
        tokens = self.tokenize(text)
        if not tokens:
            raise ValueError("示例文本没有可用词项")
        with self._lock:
            self.add_category(category_id)
            cat = self.categories[category_id]
            tf = Counter(tokens)
            cat["tf"].update(tf)
            cat["df"].update(tf.keys())
            cat["n_docs"] += 1
            self.df.update(tf.keys())
            self.n_docs += 1
        return len(tokens)

    def remove_example(self, category_id: str, text: str) -> int:
        """移除一条示例，精确抵消其贡献（换示例后边界随之调整）。"""
        tokens = self.tokenize(text)
        if not tokens:
            raise ValueError("示例文本没有可用词项")
        with self._lock:
            cat = self.categories.get(category_id)
            if not cat or cat["n_docs"] <= 0:
                raise KeyError(f"类别 {category_id} 没有可移除的示例")
            tf = Counter(tokens)
            for token, c in tf.items():
                self._decr(cat["tf"], token, c)
            for token in tf:
                self._decr(cat["df"], token)
                self._decr(self.df, token)
            cat["n_docs"] -= 1
            self.n_docs -= 1
        return len(tokens)

    def rebuild(self, examples: Iterable[tuple]) -> None:
        """从示例集合全量重建，``examples`` 为 (类别id, 类别名, 文本)。

        结果与逐条 :meth:`add_example` 的增量状态完全一致（词频为整数，
        与示例顺序无关），主要用于模型文件缺失 / 损坏时的恢复。
        """
        with self._lock:
            self.categories = {}
            self.df = Counter()
            self.n_docs = 0
            for cid, name, text in examples:
                self.add_category(cid, name)
                try:
                    self.add_example(cid, text)
                except ValueError:
                    continue  # 跳过无有效词项的示例

    # -- 分类 ---------------------------------------------------------------
    def classify(self, text: str, top_k: int = 3,
                 accept_threshold: Optional[float] = None,
                 margin: Optional[float] = None) -> dict:
        """对一篇文档分类，返回标签、置信度与候选列表。

        分类本身不修改模型，因此同一文档多次分类结果一致。
        """
        accept = self.accept_threshold if accept_threshold is None else float(accept_threshold)
        marg = self.margin if margin is None else float(margin)
        tokens = self.tokenize(text)
        if not tokens:
            raise ValueError("文本没有可用词项")
        with self._lock:
            trained = {cid: c for cid, c in self.categories.items()
                       if c["n_docs"] > 0}
            if not trained:
                raise ValueError("分类模型尚未学习任何示例，请先为类别添加示例")

            tf = Counter(tokens)
            doc_len = sum(tf.values()) or 1
            dvec = {t: (c / doc_len) * self._idf(t) for t, c in tf.items()}
            dnorm = math.sqrt(sum(v * v for v in dvec.values()))

            scores: dict[str, float] = {}
            matched: dict[str, list] = {}
            for cid in sorted(trained):  # 固定迭代顺序，保证确定性
                ctf = trained[cid]["tf"]
                # 质心 TF-IDF 向量（n_docs 缩放不影响余弦相似度）
                cnorm = math.sqrt(sum((cnt * self._idf(t)) ** 2
                                      for t, cnt in ctf.items()))
                if not cnorm or not dnorm:
                    scores[cid] = 0.0
                    matched[cid] = []
                    continue
                contrib = {t: dvec[t] * ctf[t] * self._idf(t)
                           for t in sorted(dvec) if t in ctf}
                dot = sum(contrib.values())
                scores[cid] = dot / (dnorm * cnorm)
                # 贡献最大的词项即「分类依据」
                matched[cid] = sorted(contrib,
                                      key=lambda t: (-contrib[t], t))[:5]

            # softmax 归一化为置信度
            temp = max(self.temperature, 1e-6)
            exps = {cid: math.exp(s / temp) for cid, s in scores.items()}
            z = sum(exps.values())
            conf = {cid: e / z for cid, e in exps.items()}

            order = sorted(scores,
                           key=lambda c: (-conf[c], trained[c]["name"], c))
            top = order[0]
            positive = [c for c in order if scores[c] > 0]
            accepted = bool(positive) and scores[top] >= accept
            # 置信度接近头部的类别都视为候选（边界模糊 -> 多候选）
            within = [c for c in positive if conf[top] - conf[c] <= marg]
            cand_ids = positive[:max(int(top_k), len(within))]

            candidates = [{
                "category_id": cid,
                "name": trained[cid]["name"],
                "score": round(scores[cid], 4),
                "confidence": round(conf[cid], 4),
                "matched": matched.get(cid, []),
            } for cid in cand_ids]

            return {
                "label": trained[top]["name"] if accepted else None,
                "label_id": top if accepted else None,
                "confidence": round(conf[top], 4) if cand_ids else 0.0,
                "accepted": accepted,
                "ambiguous": accepted and len(within) > 1,
                "candidates": candidates,
                "token_count": len(tokens),
                "model_fingerprint": self.fingerprint(),
            }

    # -- 状态与持久化 -------------------------------------------------------
    def stats(self) -> dict:
        """模型概览：各类别示例数、代表词项（分类依据）与指纹。"""
        with self._lock:
            cats = []
            for cid in sorted(self.categories,
                              key=lambda c: (self.categories[c]["name"], c)):
                cat = self.categories[cid]
                weighted = {t: cnt * self._idf(t) for t, cnt in cat["tf"].items()}
                top_terms = sorted(weighted,
                                   key=lambda t: (-weighted[t], t))[:8]
                cats.append({
                    "id": cid, "name": cat["name"], "n_docs": cat["n_docs"],
                    "vocab": len(cat["tf"]), "top_terms": top_terms,
                })
            return {
                "n_docs": self.n_docs,
                "vocab_size": len(self.df),
                "categories": cats,
                "params": {"temperature": self.temperature,
                           "accept_threshold": self.accept_threshold,
                           "margin": self.margin},
                "fingerprint": self.fingerprint(),
            }

    def fingerprint(self) -> str:
        """模型状态的短哈希，用于校验多次分类基于同一模型。"""
        payload = json.dumps(self._state_dict(), sort_keys=True,
                             ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def save(self, path: str) -> None:
        """原子写入模型文件（临时文件 + 替换）。"""
        with self._lock:
            data = self._state_dict()
        tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        with self._lock:
            self.categories = {}
            for cid, c in data.get("categories", {}).items():
                self.categories[cid] = {
                    "name": c.get("name", ""),
                    "tf": Counter(c.get("tf", {})),
                    "df": Counter(c.get("df", {})),
                    "n_docs": c.get("n_docs", 0),
                }
            self.df = Counter(data.get("df", {}))
            self.n_docs = data.get("n_docs", 0)

    # -- 内部 ----------------------------------------------------------------
    def _idf(self, token: str) -> float:
        return math.log((self.n_docs + 1) / (self.df.get(token, 0) + 1)) + 1.0

    @staticmethod
    def _decr(counter: Counter, key: str, by: int = 1) -> None:
        remain = counter.get(key, 0) - by
        if remain > 0:
            counter[key] = remain
        else:
            counter.pop(key, None)

    def _state_dict(self) -> dict:
        return {
            "version": self.MODEL_VERSION,
            "n_docs": self.n_docs,
            "df": dict(self.df),
            "categories": {
                cid: {"name": c["name"], "tf": dict(c["tf"]),
                      "df": dict(c["df"]), "n_docs": c["n_docs"]}
                for cid, c in self.categories.items()
            },
        }
