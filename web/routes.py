"""Flask API 路由。

把所有 NLP 能力、存储与流水线编排暴露为 REST 接口，
前端 10 个页面通过 ``fetch`` 调用这些接口。
"""

from __future__ import annotations

import json
import re
import time
import uuid
import threading
from typing import Optional

from flask import Blueprint, current_app, jsonify, request

from nlp import (get_constituency_parser, get_embeddings, get_keywords, get_ner,
                 get_parser, get_segmenter, get_sentiment, get_summarizer,
                 get_tagger, get_translator, get_classifier, ENTITY_TYPE_NAMES,
                 TAG_NAMES, DEP_REL_NAMES, PHRASE_NAMES, POLARITY_NAMES)
from nlp.classifier import IncrementalTextClassifier
from nlp.lexicon import STOPWORDS
from storage import StoreRegistry


api = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _registry() -> StoreRegistry:
    return current_app.config["STORE_REGISTRY"]


def _engine():
    return current_app.config["PIPELINE_ENGINE"]


def _models_dir() -> str:
    import os
    path = os.path.join(current_app.config["DATA_ROOT"], "models")
    os.makedirs(path, exist_ok=True)
    return path


def classifier_model_path(data_root: str) -> str:
    import os
    return os.path.join(data_root, "models", "classifier.json")


def _classifier_path() -> str:
    return classifier_model_path(current_app.config["DATA_ROOT"])


def _classifier() -> IncrementalTextClassifier:
    return get_classifier(_classifier_path())


_CLASSIFIER_LOCK = threading.RLock()


def _class_store():
    return _registry().task("classifier_class")


def _example_store():
    return _registry().task("classifier_example")


def _save_classifier() -> None:
    _classifier().save(_classifier_path())


def bootstrap_classifier(registry: StoreRegistry, data_root: str) -> IncrementalTextClassifier:
    """从磁盘载入分类模型；若已有分片但缺模型文件，则完整重建。"""
    import os
    path = classifier_model_path(data_root)
    classifier = get_classifier(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    has_classes = any(
        not r.get("_deleted") for r in registry.task("classifier_class").all()
    )
    if not os.path.exists(path) and has_classes:
        _rebuild_classifier_bootstrap(classifier, registry, path)
    return classifier


def _rebuild_classifier_bootstrap(classifier: IncrementalTextClassifier,
                                  registry: StoreRegistry, path: str) -> None:
    classifier.classes = {}
    classifier.model_version = 0
    classifier.next_class_id = 1
    max_num = 0
    for record in registry.task("classifier_class").all():
        if record.get("_deleted"):
            continue
        cid = record.get("id", "")
        classifier.add_class(
            cid, record.get("name", cid),
            description=record.get("description", ""),
            created_at=record.get("created_at"))
        if cid.startswith("class_"):
            try:
                max_num = max(max_num, int(cid.split("_", 1)[1]))
            except ValueError:
                pass
    classifier.next_class_id = max_num + 1
    classifier.fit(registry.task("classifier_example").all())
    classifier.save(path)


def _rebuild_classifier() -> None:
    """以分类/示例两个分片为事实源重建增量统计。"""
    _rebuild_classifier_bootstrap(_classifier(), _registry(), _classifier_path())


def _ensure_classifier_ready() -> Optional[tuple]:
    if len([c for c in _classifier().list_classes() if c["example_count"] > 0]) < 2:
        return jsonify({"error": "请先为至少两个类别添加示例"}), 400
    return None


def _store_result(task: str, text: str, result: dict,
                  corpus_id: Optional[str] = None) -> str:
    record = {"text": text, "result": result, "created_at": time.time()}
    if corpus_id:
        record["corpus_id"] = corpus_id
    return _registry().task(task).insert(record)


def _store_classification_result(text: str, result: dict,
                                 corpus_id: Optional[str] = None) -> str:
    record = {
        "text": text,
        "status": result["status"],
        "label_ids": result["label_ids"],
        "labels": result["labels"],
        "confidence": result["confidence"],
        "candidates": result["candidates"],
        "fit": result.get("fit"),
        "margin": result.get("margin"),
        "model_version": result["model_version"],
        "fingerprint": result["fingerprint"],
        "thresholds": result.get("thresholds", {}),
        "created_at": time.time(),
    }
    if corpus_id:
        record["corpus_id"] = corpus_id
    return _registry().task("classification").insert(record)


def _payload() -> dict:
    data = request.get_json(silent=True) or {}
    return data


def _resolve_text(data: dict) -> tuple[str, Optional[str]]:
    """从请求中取文本：优先 text，其次 corpus_id。"""
    if data.get("text"):
        return data["text"], data.get("corpus_id")
    corpus_id = data.get("corpus_id")
    if corpus_id:
        record = _registry().task("corpus").get(corpus_id)
        if record:
            return record.get("text", ""), corpus_id
        return "", corpus_id
    return "", None


def _clean(text: str, remove_stopwords: bool = True) -> dict:
    text = re.sub(r"\s+", " ", text).strip()
    seg = get_segmenter()
    words = seg.cut(text)
    if remove_stopwords:
        kept = [w for w in words if w not in STOPWORDS]
    else:
        kept = words
    removed = len(words) - len(kept)
    return {
        "text": text,
        "cleaned": " ".join(kept),
        "tokens": kept,
        "original_tokens": words,
        "removed_stopwords": removed,
    }


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------

@api.get("/status")
def status():
    return jsonify({
        "ok": True,
        "version": "1.0.0",
        "tasks": _registry().tasks(),
        "time": time.time(),
    })


@api.get("/meta")
def meta():
    """给前端提供标签集合与可配置参数。"""
    return jsonify({
        "tag_names": TAG_NAMES,
        "dep_rel_names": DEP_REL_NAMES,
        "phrase_names": PHRASE_NAMES,
        "entity_type_names": ENTITY_TYPE_NAMES,
        "polarity_names": POLARITY_NAMES,
        "directions": [{"id": "zh2en", "name": "中文 → 英文"},
                       {"id": "en2zh", "name": "英文 → 中文"}],
    })


# ---------------------------------------------------------------------------
# 语料库管理
# ---------------------------------------------------------------------------

@api.get("/corpus")
def list_corpus():
    records = _registry().task("corpus").all()
    items = [{
        "id": r.get("id"),
        "name": r.get("name", "未命名"),
        "length": len(r.get("text", "")),
        "created_at": r.get("created_at"),
        "preview": r.get("text", "")[:80],
    } for r in records if not r.get("_deleted")]
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return jsonify({"corpora": items})


@api.post("/corpus")
def create_corpus():
    data = _payload()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "语料内容不能为空"}), 400
    record = {
        "name": data.get("name") or f"语料_{int(time.time())}",
        "text": text,
        "created_at": time.time(),
    }
    rid = _registry().task("corpus").insert(record)
    return jsonify({"id": rid, "ok": True})


@api.post("/corpus/upload")
def upload_corpus():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "未接收到文件"}), 400
    raw = file.read()
    text = None
    for enc in ("utf-8", "gbk", "gb18030", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        return jsonify({"error": "无法解码文件内容"}), 400
    name = data_name = file.filename or "上传文件"
    record = {"name": name, "text": text.strip(), "created_at": time.time()}
    rid = _registry().task("corpus").insert(record)
    return jsonify({"id": rid, "name": name, "length": len(text), "ok": True})


@api.get("/corpus/<cid>")
def get_corpus(cid: str):
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    return jsonify(record)


@api.delete("/corpus/<cid>")
def delete_corpus(cid: str):
    ok = _registry().task("corpus").delete(cid)
    return jsonify({"ok": ok})


@api.post("/corpus/<cid>/clean")
def clean_corpus(cid: str):
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    data = _payload()
    result = _clean(record.get("text", ""), data.get("remove_stopwords", True))
    _store_result("clean", record.get("text", ""), result, corpus_id=cid)
    return jsonify(result)


# ---------------------------------------------------------------------------
# 分词与词性标注
# ---------------------------------------------------------------------------

@api.post("/segment")
def segment():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    seg = get_segmenter()
    words = seg.cut(text)
    result = {"words": words, "count": len(words)}
    rid = _store_result("segment", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


@api.post("/pos")
def pos_tag():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    tagger = get_tagger()
    tokens = [[w, t] for w, t in tagger.tag(text)]
    result = {"tokens": tokens, "tag_names": TAG_NAMES}
    rid = _store_result("pos", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 句法分析
# ---------------------------------------------------------------------------

@api.post("/parse")
def parse():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    dep = get_parser().parse(text)
    const = get_constituency_parser().parse(text)
    result = {
        "dependency": dep,
        "constituency": const,
        "dep_rel_names": DEP_REL_NAMES,
        "phrase_names": PHRASE_NAMES,
    }
    rid = _store_result("parse", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 命名实体识别与标注
# ---------------------------------------------------------------------------

@api.post("/ner")
def ner():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    entities = get_ner().recognize(text)
    result = {"entities": entities, "entity_type_names": ENTITY_TYPE_NAMES}
    rid = _store_result("ner", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


@api.post("/ner/annotate")
def ner_annotate():
    data = _payload()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    record = {
        "text": text,
        "entities": data.get("entities", []),
        "note": data.get("note", ""),
        "created_at": time.time(),
    }
    rid = _registry().task("annotation").insert(record)
    return jsonify({"id": rid, "ok": True})


@api.get("/ner/annotations")
def ner_annotations():
    records = _registry().task("annotation").all()
    return jsonify({"annotations": records})


# ---------------------------------------------------------------------------
# 情感分析
# ---------------------------------------------------------------------------

@api.post("/sentiment")
def sentiment():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_sentiment().analyze(text)
    result["polarity_name"] = POLARITY_NAMES.get(result["polarity"], "")
    rid = _store_result("sentiment", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 文本摘要
# ---------------------------------------------------------------------------

@api.post("/summary")
def summary():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_summarizer().summarize(
        text, ratio=data.get("ratio", 0.3),
        max_sentences=data.get("max_sentences"))
    rid = _store_result("summary", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 机器翻译（模拟）
# ---------------------------------------------------------------------------

@api.post("/translate")
def translate():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_translator().translate(text, direction=data.get("direction", "zh2en"))
    rid = _store_result("translate", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 关键词提取
# ---------------------------------------------------------------------------

@api.post("/keywords")
def keywords():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_keywords().extract(text, top_k=data.get("top_k", 10),
                                    method=data.get("method", "hybrid"))
    rid = _store_result("keywords", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 词向量
# ---------------------------------------------------------------------------

def _embedding_path() -> str:
    import os
    return os.path.join(_models_dir(), "embeddings.json")


@api.post("/embeddings/train")
def train_embeddings():
    data = _payload()
    corpus_ids = data.get("corpus_ids")
    store = _registry().task("corpus")
    if corpus_ids:
        texts = [store.get(c)["text"] for c in corpus_ids if store.get(c)]
    else:
        texts = [r["text"] for r in store.all() if not r.get("_deleted")]
    if not texts:
        return jsonify({"error": "没有可用语料，请先上传语料"}), 400

    emb = get_embeddings()
    emb.train(texts, vocab_size=data.get("vocab_size", 200),
              dim=data.get("dim", 20), window=data.get("window", 5),
              min_count=data.get("min_count", 1))

    payload = {
        "vocab": emb.vocab,
        "vectors": emb.vectors,
        "dim": emb.dim,
        "trained_at": time.time(),
        "corpus_count": len(texts),
    }
    with open(_embedding_path(), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return jsonify(emb.stats())


@api.get("/embeddings/vectors")
def embeddings_vectors():
    emb = get_embeddings()
    if not emb.vectors:
        _load_embeddings()
        emb = get_embeddings()
    if not emb.vectors:
        return jsonify({"error": "尚未训练词向量"}), 404
    n_clusters = int(request.args.get("clusters", 5))
    proj = emb.project_2d()
    clusters = emb.cluster(n_clusters)
    return jsonify({
        "points": [{"word": w, "x": round(p[0], 4), "y": round(p[1], 4),
                    "cluster": clusters.get(w, 0)} for w, p in proj.items()],
        "stats": emb.stats(),
    })


@api.get("/embeddings/neighbors")
def embeddings_neighbors():
    word = request.args.get("word", "")
    k = int(request.args.get("k", 10))
    emb = get_embeddings()
    if not emb.vectors:
        _load_embeddings()
        emb = get_embeddings()
    return jsonify({"word": word, "neighbors": emb.nearest(word, k)})


def _load_embeddings():
    import os
    path = _embedding_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        emb = get_embeddings()
        emb.vocab = data.get("vocab", [])
        emb.vectors = data.get("vectors", {})
        emb.dim = data.get("dim", 0)
    except (json.JSONDecodeError, OSError):
        pass


# ---------------------------------------------------------------------------
# 自动文本分类（增量 TF-IDF 质心）
# ---------------------------------------------------------------------------

def _classification_thresholds(data: dict) -> dict:
    keys = ("accept_confidence", "review_confidence", "ambiguous_confidence",
            "min_fit", "margin", "ratio", "temperature")
    result = {}
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if key == "temperature":
            if 0.01 <= number <= 1.0:
                result[key] = number
        elif 0.0 <= number <= 1.0:
            result[key] = number
    return result


@api.get("/classifier/classes")
def classifier_classes():
    classifier = _classifier()
    return jsonify({
        "classes": classifier.list_classes(),
        "profile": classifier.profile(),
        "stats": classifier.stats(),
    })


@api.post("/classifier/classes")
def create_classifier_class():
    data = _payload()
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "类别名称不能为空"}), 400
    with _CLASSIFIER_LOCK:
        classifier = _classifier()
        cid = classifier.reserve_class_id()
        record = {
            "id": cid,
            "name": name,
            "description": (data.get("description") or "").strip(),
            "created_at": time.time(),
        }
        _class_store().insert(record)
        classifier.add_class(cid, name, record["description"], record["created_at"])
        _save_classifier()
        return jsonify({"id": cid, "class": {
            "id": cid,
            "name": name,
            "description": record["description"],
            "example_count": 0,
            "created_at": record["created_at"],
        }, "ok": True})


@api.put("/classifier/classes/<cid>")
def update_classifier_class(cid: str):
    data = _payload()
    with _CLASSIFIER_LOCK:
        class_record = _class_store().get(cid)
        if not class_record or class_record.get("_deleted"):
            return jsonify({"error": "类别不存在"}), 404
        name = (data.get("name") or class_record.get("name", cid)).strip()
        description = (data.get("description")
                       if data.get("description") is not None
                       else class_record.get("description", ""))
        if not name:
            return jsonify({"error": "类别名称不能为空"}), 400
        _class_store().update(cid, {
            "name": name,
            "description": description,
            "updated_at": time.time(),
        })
        _classifier().update_class(cid, name=name, description=description)
        _save_classifier()
        return jsonify({"id": cid, "ok": True})


@api.delete("/classifier/classes/<cid>")
def delete_classifier_class(cid: str):
    with _CLASSIFIER_LOCK:
        class_record = _class_store().get(cid)
        if not class_record or class_record.get("_deleted"):
            return jsonify({"error": "类别不存在"}), 404
        classifier = _classifier()
        examples = _example_store().all()
        removed_examples = 0
        for example in examples:
            if example.get("_deleted") or cid not in example.get("label_ids", []):
                continue
            labels = [label for label in example.get("label_ids", []) if label != cid]
            if labels:
                terms = example.get("terms", [])
                classifier.remove_document(terms, example.get("label_ids", []))
                label_names = [classifier.classes[label]["name"] for label in labels]
                classifier.add_document(example.get("text", ""), labels, terms=terms)
                _example_store().update(example["id"], {
                    "label_ids": labels,
                    "labels": label_names,
                    "updated_at": time.time(),
                })
            else:
                classifier.remove_document(example.get("terms", []),
                                           example.get("label_ids", []))
                _example_store().delete(example["id"])
            removed_examples += 1
        classifier.remove_class(cid)
        _class_store().delete(cid)
        _save_classifier()
        return jsonify({"ok": True, "removed_examples": removed_examples})


@api.get("/classifier/examples")
def list_classifier_examples():
    cid = request.args.get("class_id")
    where = [("label_ids", "contains", cid)] if cid else None
    records = _example_store().query(where=where, order_by="_created", order="desc")
    return jsonify({"examples": records})


@api.post("/classifier/examples")
def add_classifier_example():
    data = _payload()
    text = (data.get("text") or "").strip()
    label_ids = data.get("label_ids") or []
    if isinstance(label_ids, str):
        label_ids = [label_ids]
    if not text:
        return jsonify({"error": "示例文本不能为空"}), 400
    if not label_ids:
        return jsonify({"error": "请至少选择一个类别"}), 400

    with _CLASSIFIER_LOCK:
        classifier = _classifier()
        missing = [cid for cid in label_ids if cid not in classifier.classes]
        if missing:
            return jsonify({"error": f"类别不存在: {', '.join(missing)}"}), 400
        terms = classifier.tokenize(text)
        if not terms:
            return jsonify({"error": "示例没有可用特征词，请补充更有内容的文本"}), 400
        label_names = [classifier.classes[cid]["name"] for cid in label_ids]
        record = {
            "text": text,
            "label_ids": label_ids,
            "labels": label_names,
            "terms": terms,
            "source": data.get("source", "manual"),
            "created_at": time.time(),
        }
        eid = _example_store().insert(record)
        classifier.add_document(text, label_ids, terms=terms)
        _save_classifier()
        record["id"] = eid
        return jsonify({"id": eid, "example": record,
                        "stats": classifier.stats(), "ok": True})


@api.patch("/classifier/examples/<eid>")
def update_classifier_example(eid: str):
    data = _payload()
    with _CLASSIFIER_LOCK:
        record = _example_store().get(eid)
        if not record or record.get("_deleted"):
            return jsonify({"error": "示例不存在"}), 404
        classifier = _classifier()
        classifier.remove_document(record.get("terms", []), record.get("label_ids", []))

        text = (data.get("text") or record.get("text", "")).strip()
        label_ids = data.get("label_ids") or record.get("label_ids", [])
        if isinstance(label_ids, str):
            label_ids = [label_ids]
        missing = [cid for cid in label_ids if cid not in classifier.classes]
        if missing:
            # 回滚内存状态，避免一次失败请求改变模型
            classifier.add_document(record.get("text", ""),
                                    record.get("label_ids", []),
                                    terms=record.get("terms", []))
            return jsonify({"error": f"类别不存在: {', '.join(missing)}"}), 400
        terms = classifier.tokenize(text)
        if not terms:
            classifier.add_document(record.get("text", ""),
                                    record.get("label_ids", []),
                                    terms=record.get("terms", []))
            return jsonify({"error": "示例没有可用特征词"}), 400
        label_names = [classifier.classes[cid]["name"] for cid in label_ids]
        _example_store().update(eid, {
            "text": text,
            "label_ids": label_ids,
            "labels": label_names,
            "terms": terms,
            "updated_at": time.time(),
        })
        classifier.add_document(text, label_ids, terms=terms)
        _save_classifier()
        return jsonify({"id": eid, "ok": True})


@api.delete("/classifier/examples/<eid>")
def delete_classifier_example(eid: str):
    with _CLASSIFIER_LOCK:
        record = _example_store().get(eid)
        if not record or record.get("_deleted"):
            return jsonify({"error": "示例不存在"}), 404
        _classifier().remove_document(record.get("terms", []),
                                      record.get("label_ids", []))
        _example_store().delete(eid)
        _save_classifier()
        return jsonify({"ok": True, "stats": _classifier().stats()})


def _run_classification(text: str, data: dict, persist: bool = True) -> dict:
    thresholds = _classification_thresholds(data)
    with _CLASSIFIER_LOCK:
        classifier = _classifier()
        result = classifier.classify(text, thresholds=thresholds)
        if persist:
            result["id"] = _store_classification_result(
                text, result, corpus_id=data.get("corpus_id"))
        return result


@api.post("/classifier/classify")
def classify_text():
    not_ready = _ensure_classifier_ready()
    if not_ready:
        return not_ready
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少待分类文本"}), 400
    data = dict(data)
    data["corpus_id"] = cid
    try:
        return jsonify(_run_classification(text, data))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@api.post("/classifier/classify-batch")
def classify_corpus_batch():
    not_ready = _ensure_classifier_ready()
    if not_ready:
        return not_ready
    data = _payload()
    store = _registry().task("corpus")
    corpus_ids = data.get("corpus_ids") or []
    if corpus_ids:
        documents = []
        for cid in corpus_ids:
            record = store.get(cid)
            if record and not record.get("_deleted"):
                documents.append((cid, record.get("text", "")))
    else:
        documents = [(r["id"], r.get("text", "")) for r in store.all()
                     if not r.get("_deleted")]
    if not documents:
        return jsonify({"error": "没有可分类的文档"}), 400
    results = []
    for cid, text in documents:
        payload = dict(data)
        payload["corpus_id"] = cid
        results.append({
            "corpus_id": cid,
            "preview": text[:120],
            "result": _run_classification(text, payload),
        })
    return jsonify({"count": len(results), "results": results})


@api.get("/classifier/model")
def classifier_model():
    classifier = _classifier()
    return jsonify({"stats": classifier.stats(), "profile": classifier.profile()})


@api.post("/classifier/rebuild")
def rebuild_classifier_model():
    with _CLASSIFIER_LOCK:
        _rebuild_classifier()
        return jsonify({"ok": True, "stats": _classifier().stats()})


# ---------------------------------------------------------------------------
# 流水线配置与执行
# ---------------------------------------------------------------------------

@api.get("/pipeline/stages")
def pipeline_stages():
    return jsonify({"stages": _engine().list_stages()})


@api.post("/pipeline")
def save_pipeline():
    data = _payload()
    config = data.get("config") or data
    if not config.get("stages"):
        return jsonify({"error": "流水线至少需要一个阶段"}), 400
    name = config.get("name") or f"流水线_{int(time.time())}"
    record = {"name": name, "config": config, "created_at": time.time()}
    rid = _registry().task("pipeline_config").insert(record)
    return jsonify({"id": rid, "name": name, "ok": True})


@api.get("/pipeline")
def list_pipelines():
    records = _registry().task("pipeline_config").all()
    items = [{"id": r["id"], "name": r.get("name"), "config": r.get("config"),
              "created_at": r.get("created_at")}
             for r in records if not r.get("_deleted")]
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return jsonify({"pipelines": items})


@api.get("/pipeline/<pid>")
def get_pipeline(pid: str):
    record = _registry().task("pipeline_config").get(pid)
    if not record:
        return jsonify({"error": "流水线不存在"}), 404
    return jsonify(record)


@api.post("/pipeline/preview")
def pipeline_preview():
    """对单条文本跑流水线（不持久化），供配置页预览。"""
    data = _payload()
    text = (data.get("text") or "").strip()
    config = data.get("config")
    if not text or not config:
        return jsonify({"error": "缺少文本或配置"}), 400
    try:
        result = _engine().build(config).run({
            "text": text,
            "classifier_path": _classifier_path(),
        })
        return jsonify({"ok": True, "output": result})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@api.post("/pipeline/<pid>/run")
def run_pipeline(pid: str):
    record = _registry().task("pipeline_config").get(pid)
    if not record:
        return jsonify({"error": "流水线不存在"}), 404
    config = record.get("config")
    data = _payload()

    run_id = uuid.uuid4().hex[:12]
    started = time.time()

    if data.get("batch"):
        # 批量：对语料库中的多篇文档执行
        corpus_ids = data.get("corpus_ids") or []
        store = _registry().task("corpus")
        docs = []
        if corpus_ids:
            docs = [store.get(c)["text"] for c in corpus_ids if store.get(c)]
        else:
            docs = [r["text"] for r in store.all() if not r.get("_deleted")]
        if not docs:
            return jsonify({"error": "没有可处理的文档"}), 400

        progress_state = {"done": 0, "total": len(docs)}

        def _progress(done, total):
            progress_state["done"] = done
            progress_state["total"] = total

        shared = dict(data.get("shared") or {})
        shared.setdefault("classifier_path", _classifier_path())
        results = _engine().run_batch(
            config, docs, shared=shared,
            max_workers=data.get("max_workers", 4),
            chunk_size=data.get("chunk_size", 16),
            progress=_progress)
        succeeded = sum(1 for r in results if r and r["ok"])
        failed = len(results) - succeeded
        run_record = {
            "run_id": run_id, "pipeline_id": pid, "batch": True,
            "doc_count": len(docs), "succeeded": succeeded, "failed": failed,
            "started": started, "finished": time.time(),
            "results": results,
        }
        rid = _registry().task("pipeline_run").insert(run_record)
        return jsonify({"run_id": run_id, "id": rid, "succeeded": succeeded,
                        "failed": failed, "doc_count": len(docs)})
    else:
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "缺少文本"}), 400
        try:
            output = _engine().build(config).run({
                "text": text,
                "classifier_path": _classifier_path(),
            })
            run_record = {
                "run_id": run_id, "pipeline_id": pid, "batch": False,
                "text": text, "output": output,
                "started": started, "finished": time.time(),
            }
            rid = _registry().task("pipeline_run").insert(run_record)
            return jsonify({"run_id": run_id, "id": rid, "ok": True,
                            "output": output})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 400


@api.get("/pipeline/run/<run_id>")
def get_pipeline_run(run_id: str):
    records = _registry().task("pipeline_run").query(
        where=[("run_id", "eq", run_id)])
    if not records:
        return jsonify({"error": "执行记录不存在"}), 404
    return jsonify(records[0])


# ---------------------------------------------------------------------------
# 结果查询（分片合并与查询）
# ---------------------------------------------------------------------------

@api.get("/results")
def list_result_tasks():
    registry = _registry()
    tasks = []
    for name in registry.tasks():
        if name in ("corpus", "pipeline_config", "annotation",
                    "classifier_class", "classifier_example"):
            continue
        stats = registry.task(name).stats()
        tasks.append(stats)
    return jsonify({"tasks": tasks})


@api.get("/results/<task>")
def query_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    store = registry.task(task)
    where = []
    for key in ("type", "corpus_id"):
        val = request.args.get(key)
        if val:
            where.append((key, "eq", val))
    order_by = request.args.get("order_by")
    order = request.args.get("order", "desc")
    limit = request.args.get("limit", type=int)
    offset = request.args.get("offset", 0, type=int)
    records = store.query(where=where or None, order_by=order_by,
                          order=order, limit=limit, offset=offset)
    return jsonify({
        "task": task,
        "count": len(records),
        "stats": store.stats(),
        "records": records,
    })


@api.post("/results/<task>/compact")
def compact_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(registry.task(task).compact())


@api.get("/results/<task>/merge")
def merge_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(registry.task(task).merge())


@api.post("/results/compact_all")
def compact_all():
    return jsonify({"compacted": _registry().compact_all()})
