"""少样本文本增量分类器。

算法采用确定性的「增量 TF-IDF + 类别质心」：

* 每个类别维护该类所有示例的亚线性词频累加和，新增/删除示例时只更新
  倒排文档频率（DF）和对应类别向量，不需要全库重学；
* 预测时用当前 DF 现场计算 IDF，因此更换、增加示例后类别边界会稳定调整；
* 文档与类别质心都做 L2 归一化，用余弦相似度衡量归属；
* 相似度经 softmax 校准为置信度，并结合最高相似度、候选间隔、相似度比例
  区分「确定」「多候选/边界模糊」和「需人工复核」；
* 全过程没有随机初始化或采样，同一模型版本下同一文本结果可复现。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from typing import Iterable, Optional

from .lexicon import STOPWORDS
from .segmenter import Segmenter


def _atomic_write_json(path: str, obj: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".classifier-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class IncrementalTextClassifier:
    """持久化、可增量更新的少样本文本分类器。"""

    def __init__(self, path: Optional[str] = None, segmenter: Optional[Segmenter] = None):
        self.path = path
        self.segmenter = segmenter or Segmenter()
        self.model_version = 0
        self.next_class_id = 1
        self.document_count = 0
        self.classes: dict[str, dict] = {}
        self.df: dict[str, int] = {}
        self.class_counts: dict[str, int] = {}
        self.class_term_weights: dict[str, dict[str, float]] = {}
        self.updated_at: Optional[float] = None
        if path and os.path.exists(path):
            self.load(path)

    # ------------------------------------------------------------------
    # 文本向量化
    # ------------------------------------------------------------------
    def tokenize(self, text: str) -> list[str]:
        """分词并过滤标点、停用词；英文统一小写。"""
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            return []
        tokens = self.segmenter.cut(text)
        result = []
        for token in tokens:
            token = token.strip().lower()
            if not token or token in STOPWORDS:
                continue
            if not re.search(r"[一-鿿a-zA-Z0-9]", token):
                continue
            # 保留英文/数字词；中文过滤信息量较低的单字。
            if len(token) < 2 and not re.search(r"[a-zA-Z0-9]", token):
                continue
            result.append(token)
        return result

    @staticmethod
    def term_frequency(terms: Iterable[str]) -> dict[str, float]:
        """亚线性词频：1 + log(tf)，降低长文档和重复词的支配作用。"""
        counts = Counter(terms)
        return {term: 1.0 + math.log(count) for term, count in counts.items()}

    def _idf(self, term: str) -> float:
        # 平滑 IDF；未见词不会产生除零，且因质心无该词而拉低整体拟合度。
        df = self.df.get(term, 0)
        return math.log((self.document_count + 1.0) / (df + 1.0)) + 1.0

    # ------------------------------------------------------------------
    # 类别管理
    # ------------------------------------------------------------------
    def list_classes(self) -> list[dict]:
        items = []
        for cid, info in self.classes.items():
            items.append({
                "id": cid,
                "name": info.get("name", cid),
                "description": info.get("description", ""),
                "example_count": self.class_counts.get(cid, 0),
                "created_at": info.get("created_at"),
                "updated_at": info.get("updated_at"),
            })
        items.sort(key=lambda x: x["id"])
        return items

    def reserve_class_id(self) -> str:
        cid = f"class_{self.next_class_id}"
        self.next_class_id += 1
        return cid

    def add_class(self, cid: str, name: str, description: str = "",
                  created_at: Optional[float] = None) -> str:
        if cid in self.classes:
            raise ValueError(f"类别已存在: {cid}")
        now = created_at or time.time()
        self.classes[cid] = {
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
        }
        self.class_counts.setdefault(cid, 0)
        self.class_term_weights.setdefault(cid, {})
        self._bump_version()
        return cid

    def update_class(self, cid: str, name: Optional[str] = None,
                     description: Optional[str] = None) -> None:
        if cid not in self.classes:
            raise ValueError(f"类别不存在: {cid}")
        changed = False
        if name is not None and name != self.classes[cid].get("name"):
            self.classes[cid]["name"] = name
            changed = True
        if description is not None and description != self.classes[cid].get("description"):
            self.classes[cid]["description"] = description
            changed = True
        if changed:
            self.classes[cid]["updated_at"] = time.time()
            self._bump_version()

    def remove_class(self, cid: str) -> None:
        self.classes.pop(cid, None)
        self.class_counts.pop(cid, None)
        self.class_term_weights.pop(cid, None)
        self._bump_version()

    # ------------------------------------------------------------------
    # 增量学习
    # ------------------------------------------------------------------
    def add_document(self, text: str, label_ids: list[str],
                     terms: Optional[list[str]] = None) -> list[str]:
        """吸收一个可多标签示例，返回实际使用的特征词。"""
        terms = terms if terms is not None else self.tokenize(text)
        self._add_terms(terms, label_ids)
        self._bump_version()
        return terms

    def add_documents(self, records: list[dict]) -> None:
        """批量吸收示例，只把模型版本推进一次。"""
        if not records:
            return
        for record in records:
            terms = record.get("terms")
            if terms is None:
                terms = self.tokenize(record.get("text", ""))
            self._add_terms(terms, record.get("label_ids", []))
        self._bump_version()

    def _add_terms(self, terms: list[str], label_ids: list[str]) -> None:
        labels = self._validate_labels(label_ids)
        if not terms:
            raise ValueError("示例没有可用特征词，请提供更有内容的文本")
        weights = self.term_frequency(terms)
        for term in weights:
            self.df[term] = self.df.get(term, 0) + 1
        self.document_count += 1
        for cid in labels:
            self.class_counts[cid] = self.class_counts.get(cid, 0) + 1
            bucket = self.class_term_weights.setdefault(cid, {})
            for term, weight in weights.items():
                bucket[term] = bucket.get(term, 0.0) + weight

    def remove_document(self, terms: list[str], label_ids: list[str]) -> None:
        """删除一个示例并递减增量统计；空类别保留，边界重新计算。"""
        labels = [cid for cid in label_ids if cid in self.classes]
        weights = self.term_frequency(terms)
        for term in weights:
            if term in self.df:
                self.df[term] -= 1
                if self.df[term] <= 0:
                    self.df.pop(term, None)
        self.document_count = max(0, self.document_count - 1)
        for cid in labels:
            self.class_counts[cid] = max(0, self.class_counts.get(cid, 0) - 1)
            bucket = self.class_term_weights.get(cid, {})
            for term, weight in weights.items():
                bucket[term] = bucket.get(term, 0.0) - weight
                if bucket[term] <= 1e-12:
                    bucket.pop(term, None)
        self._bump_version()

    def _validate_labels(self, label_ids: list[str]) -> list[str]:
        if not label_ids:
            raise ValueError("至少需要一个类别标签")
        labels = []
        for cid in label_ids:
            if cid not in self.classes:
                raise ValueError(f"类别不存在: {cid}")
            if cid not in labels:
                labels.append(cid)
        return labels

    def fit(self, records: list[dict]) -> "IncrementalTextClassifier":
        """从示例分片完整重建模型。"""
        self.model_version = 0
        self.document_count = 0
        self.df = {}
        self.class_counts = {cid: 0 for cid in self.classes}
        self.class_term_weights = {cid: {} for cid in self.classes}
        active = set(self.classes)
        batch = []
        for record in records:
            if record.get("_deleted"):
                continue
            labels = [cid for cid in record.get("label_ids", []) if cid in active]
            if not labels:
                continue
            terms = record.get("terms")
            if terms is None:
                terms = self.tokenize(record.get("text", ""))
            if terms:
                batch.append({"terms": terms, "label_ids": labels})
        self.add_documents(batch)
        return self

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def classify(self, text: str, thresholds: Optional[dict] = None) -> dict:
        terms = self.tokenize(text)
        return self.classify_terms(terms, thresholds=thresholds, source_text=text)

    def classify_terms(self, terms: list[str], thresholds: Optional[dict] = None,
                       source_text: str = "") -> dict:
        cfg = {
            "accept_confidence": 0.68,
            "review_confidence": 0.55,
            "ambiguous_confidence": 0.40,
            "min_fit": 0.18,
            "margin": 0.10,
            "ratio": 0.70,
            "temperature": 0.12,
        }
        cfg.update(thresholds or {})

        usable = [cid for cid, count in self.class_counts.items()
                  if cid in self.classes and count > 0]
        if len(usable) < 2:
            raise ValueError("至少需要两个包含示例的类别才能分类")
        if not terms:
            return self._empty_result(source_text, terms, cfg)

        query = self._vector(terms)
        q_norm = math.sqrt(sum(v * v for v in query.values())) or 1.0
        raw_scores: list[tuple[str, float, float]] = []
        query_fit = math.sqrt(sum(
            w * w for term, w in query.items() if self.df.get(term, 0) > 0
        )) / q_norm
        for cid in usable:
            centroid, c_norm = self._centroid(cid)
            dot = sum(weight * centroid.get(term, 0.0)
                      for term, weight in query.items())
            cosine = dot / (q_norm * c_norm) if c_norm else 0.0
            raw_scores.append((cid, max(0.0, cosine), query_fit))

        fit = sum(f for _, _, f in raw_scores) / len(raw_scores) if raw_scores else 0.0
        raw_scores.sort(key=lambda item: (-item[1], -item[2], item[0]))

        exp_scores = [math.exp(max(item[1], 0.0) / cfg["temperature"]) for item in raw_scores]
        total = sum(exp_scores) or 1.0
        softmax_scores = [value / total for value in exp_scores]
        # softmax 在多候选时可能仍偏尖，平方根校准让高相似候选能共同进入候选。
        confidences = [math.sqrt(value) for value in softmax_scores]
        confidence_sum = sum(confidences) or 1.0
        confidences = [value / confidence_sum for value in confidences]

        top_cosine = raw_scores[0][1] if raw_scores else 0.0
        second_cosine = raw_scores[1][1] if len(raw_scores) > 1 else 0.0
        margin = top_cosine - second_cosine
        ratio = second_cosine / top_cosine if top_cosine > 1e-12 else 0.0
        candidates = []
        for index, (cid, cosine, _) in enumerate(raw_scores):
            candidates.append({
                "class_id": cid,
                "class_name": self.classes[cid]["name"],
                "similarity": round(cosine, 4),
                "confidence": round(confidences[index], 4),
            })

        chosen = candidates[:1]
        if candidates:
            top_similarity = candidates[0]["similarity"]
            for candidate in candidates[1:]:
                close_by_margin = top_similarity - candidate["similarity"] <= cfg["margin"]
                close_by_ratio = (
                    top_similarity > 1e-12
                    and candidate["similarity"] / top_similarity >= cfg["ratio"]
                )
                if (candidate["confidence"] >= cfg["ambiguous_confidence"]
                        and (close_by_margin or close_by_ratio)):
                    chosen.append(candidate)
        top_confidence = candidates[0]["confidence"] if candidates else 0.0
        top_accepted = (
            top_confidence >= cfg["accept_confidence"]
            and top_cosine >= cfg["min_fit"]
        )

        if fit < cfg["min_fit"]:
            status = "review"
            label_ids = []
        elif len(chosen) > 1:
            second_accepted = (
                chosen[1]["confidence"] >= cfg["ambiguous_confidence"]
                and chosen[1]["similarity"] >= cfg["min_fit"]
            )
            if top_accepted and second_accepted:
                status = "multi_label"
                label_ids = [item["class_id"] for item in chosen]
            else:
                status = "ambiguous"
                label_ids = []
        elif top_accepted:
            status = "confident"
            label_ids = [chosen[0]["class_id"]]
        elif top_confidence >= cfg["review_confidence"] and top_cosine >= cfg["min_fit"]:
            status = "ambiguous"
            label_ids = []
        else:
            status = "review"
            label_ids = []

        fingerprint = self._fingerprint(source_text, cfg)
        return {
            "status": status,
            "label_ids": label_ids,
            "labels": [self.classes[cid]["name"] for cid in label_ids],
            "confidence": round(top_confidence, 4),
            "fit": round(fit, 4),
            "margin": round(margin, 4),
            "candidates": candidates,
            "terms": terms,
            "model_version": self.model_version,
            "fingerprint": fingerprint,
            "thresholds": {key: cfg[key] for key in (
                "accept_confidence", "review_confidence", "ambiguous_confidence",
                "min_fit", "margin", "ratio", "temperature")},
        }

    def _empty_result(self, text: str, terms: list[str], cfg: dict) -> dict:
        return {
            "status": "review",
            "label_ids": [],
            "labels": [],
            "confidence": 0.0,
            "fit": 0.0,
            "margin": 0.0,
            "candidates": [],
            "terms": terms,
            "model_version": self.model_version,
            "fingerprint": self._fingerprint(text, cfg),
            "thresholds": {key: cfg[key] for key in (
                "accept_confidence", "review_confidence", "ambiguous_confidence",
                "min_fit", "margin", "ratio", "temperature")},
            "message": "文本中没有足够的有效特征词",
        }

    def _vector(self, terms: list[str]) -> dict[str, float]:
        tf = self.term_frequency(terms)
        return {term: weight * self._idf(term) for term, weight in tf.items()}

    def _centroid(self, class_id: str) -> tuple[dict[str, float], float]:
        count = self.class_counts.get(class_id, 0)
        if count <= 0:
            return {}, 0.0
        raw = self.class_term_weights.get(class_id, {})
        vector = {}
        for term, total in raw.items():
            value = (total / count) * self._idf(term)
            if abs(value) > 1e-12:
                vector[term] = value
        norm = math.sqrt(sum(v * v for v in vector.values()))
        return vector, norm or 1.0

    def _fingerprint(self, text: str, thresholds: dict) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        payload = json.dumps({
            "text": normalized,
            "version": self.model_version,
            "thresholds": thresholds,
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # 可解释性与持久化
    # ------------------------------------------------------------------
    def profile(self, top_k: int = 10) -> list[dict]:
        result = []
        for cid in sorted(self.classes):
            centroid, _ = self._centroid(cid)
            terms = sorted(centroid.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            result.append({
                "class_id": cid,
                "class_name": self.classes[cid]["name"],
                "example_count": self.class_counts.get(cid, 0),
                "top_terms": [{"term": term, "weight": round(weight, 4)}
                              for term, weight in terms],
            })
        return result

    def stats(self) -> dict:
        return {
            "model_version": self.model_version,
            "document_count": self.document_count,
            "class_count": len(self.classes),
            "active_class_count": sum(1 for c in self.class_counts.values() if c > 0),
            "vocab_size": len(self.df),
            "updated_at": self.updated_at,
            "classes": self.list_classes(),
        }

    def to_dict(self) -> dict:
        return {
            "format": "incremental-tfidf-centroid-v1",
            "model_version": self.model_version,
            "next_class_id": self.next_class_id,
            "document_count": self.document_count,
            "updated_at": self.updated_at,
            "classes": self.classes,
            "df": self.df,
            "class_counts": self.class_counts,
            "class_term_weights": self.class_term_weights,
        }

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path:
            raise ValueError("未指定模型持久化路径")
        self.path = path
        _atomic_write_json(path, self.to_dict())

    def load(self, path: str) -> "IncrementalTextClassifier":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.path = path
        self.model_version = int(data.get("model_version", 0))
        self.next_class_id = int(data.get("next_class_id", 1))
        self.document_count = int(data.get("document_count", 0))
        self.updated_at = data.get("updated_at")
        self.classes = data.get("classes", {})
        self.df = {k: int(v) for k, v in data.get("df", {}).items()}
        self.class_counts = {k: int(v) for k, v in data.get("class_counts", {}).items()}
        self.class_term_weights = {
            cid: {term: float(weight) for term, weight in terms.items()}
            for cid, terms in data.get("class_term_weights", {}).items()
        }
        for cid in self.classes:
            self.class_counts.setdefault(cid, 0)
            self.class_term_weights.setdefault(cid, {})
        return self

    def _bump_version(self) -> None:
        self.model_version += 1
        self.updated_at = time.time()
