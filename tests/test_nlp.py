"""NLP 平台单元测试。

运行：``python -m unittest discover -s tests -v``
覆盖：分词、词性、句法、NER、情感、摘要、翻译、关键词、词向量、
分片存储（含并发锁）、流水线引擎、HMM。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import unittest

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import (get_segmenter, get_tagger, get_parser, get_constituency_parser,
                 get_ner, get_sentiment, get_summarizer, get_translator,
                 get_keywords, get_embeddings, TAGSET)
from nlp.classifier import IncrementalTextClassifier
from nlp.hmm import HMM
from pipeline import PipelineEngine, PipelineError
from storage import ShardedStore, StoreRegistry

try:
    from app import create_app
except ImportError:  # Flask 是 Web 运行时依赖，纯算法测试环境可不安装
    create_app = None


class TestSegmenter(unittest.TestCase):
    def test_basic(self):
        words = get_segmenter().cut("自然语言处理是人工智能的重要分支")
        self.assertIn("自然语言", words)
        self.assertIn("人工智能", words)
        self.assertIn("是", words)

    def test_english_number(self):
        words = get_segmenter().cut("我用Python写了100行代码")
        self.assertIn("Python", words)
        self.assertIn("100", words)


class TestPOSTagger(unittest.TestCase):
    def test_tags(self):
        pairs = get_tagger().tag("我学习自然语言处理")
        self.assertTrue(pairs)
        for word, tag in pairs:
            self.assertIn(tag, TAGSET, f"{word}:{tag}")

    def test_punct_as_other(self):
        pairs = get_tagger().tag("你好，世界。")
        tags = [t for _, t in pairs]
        for t in tags:
            if t in ("，", "。"):
                continue
        # 标点词本身应为 x
        for w, t in pairs:
            if w in ("，", "。"):
                self.assertEqual(t, "x")


class TestParser(unittest.TestCase):
    def test_dependency(self):
        dep = get_parser().parse("北京大学的研究团队开发了机器学习系统")
        self.assertEqual(len(dep["words"]), len(dep["heads"]))
        self.assertIn(-1, dep["heads"])  # 存在根
        # 每个 head 都是有效下标或 -1
        for h in dep["heads"]:
            self.assertTrue(h == -1 or 0 <= h < len(dep["words"]))

    def test_constituency_spans(self):
        c = get_constituency_parser().parse("北京大学的研究团队开发了系统")
        leaves = self._leaves(c["tree"])
        self.assertEqual("北京大学的研究团队开发了系统", leaves)

    @staticmethod
    def _leaves(tree):
        if not tree.get("children"):
            return tree.get("word", "")
        return "".join(TestParser._leaves(ch) for ch in tree["children"])


class TestNER(unittest.TestCase):
    def test_known_entities(self):
        ents = get_ner().recognize("马云在北京工作")
        types = {e["text"]: e["type"] for e in ents}
        self.assertEqual(types.get("马云"), "PERSON")
        self.assertEqual(types.get("北京"), "LOCATION")

    def test_date_money(self):
        ents = get_ner().recognize("2024年10月1日花了99.9元")
        texts = [e["text"] for e in ents]
        self.assertTrue(any("2024" in t for t in texts))
        self.assertTrue(any("99.9" in t for t in texts))


class TestSentiment(unittest.TestCase):
    def test_positive(self):
        r = get_sentiment().analyze("这个产品非常好用，我很喜欢")
        self.assertEqual(r["polarity"], "positive")

    def test_negative(self):
        r = get_sentiment().analyze("服务态度很差，令人失望")
        self.assertEqual(r["polarity"], "negative")


class TestSummarizer(unittest.TestCase):
    def test_shorter(self):
        text = ("自然语言处理是人工智能的重要分支。它研究如何让计算机理解语言。"
                "分词是基础任务。词性标注是另一个任务。")
        r = get_summarizer().summarize(text, ratio=0.5)
        self.assertTrue(len(r["summary"]) < len(text))
        self.assertTrue(r["top_indices"])


class TestTranslator(unittest.TestCase):
    def test_zh2en(self):
        r = get_translator().translate("我喜欢机器学习", "zh2en")
        self.assertIn("machine learning", r["translation"].lower())

    def test_en2zh(self):
        r = get_translator().translate("I like China", "en2zh")
        self.assertTrue(r["translation"])


class TestKeywords(unittest.TestCase):
    def test_extract(self):
        r = get_keywords().extract("自然语言处理是人工智能的重要分支", top_k=5)
        self.assertTrue(r["keywords"])
        for k in r["keywords"]:
            self.assertIn("word", k)
            self.assertIn("score", k)


class TestEmbeddings(unittest.TestCase):
    def test_train_nearest(self):
        texts = [
            "自然语言处理是人工智能的重要分支",
            "机器学习是人工智能的核心技术",
            "深度学习推动了人工智能的发展",
            "分词是自然语言处理的基础任务",
        ] * 3
        emb = get_embeddings()
        emb.train(texts, vocab_size=60, dim=8, window=3, min_count=1)
        self.assertTrue(emb.vocab)
        self.assertTrue(emb.vectors)
        # 近邻应返回词且不包含自身
        nb = emb.nearest(emb.vocab[0], k=3)
        self.assertTrue(nb)
        self.assertNotIn(emb.vocab[0], [n["word"] for n in nb])
        # 2D 投影
        proj = emb.project_2d()
        self.assertEqual(len(proj), len(emb.vectors))


class TestIncrementalClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = IncrementalTextClassifier()
        self.clf.add_class("class_1", "财务")
        self.clf.add_class("class_2", "技术")
        self.finance = [
            "财务部完成预算审计，员工提交发票报销和税务申报材料",
            "公司确认营业收入和现金流，核算成本利润并完成付款审批",
            "审计人员检查报销单、发票、预算执行情况和财务报表",
        ]
        self.tech = [
            "算法团队训练机器学习模型，在服务器上优化推理性能",
            "后端工程师发布微服务接口，修复系统漏洞并完善监控",
            "软件开发人员重构数据结构，编写自动化测试和部署脚本",
        ]
        for text in self.finance:
            self.clf.add_document(text, ["class_1"])
        for text in self.tech:
            self.clf.add_document(text, ["class_2"])

    def test_confident_deterministic(self):
        text = "请提交发票和报销单，财务部门完成预算审计"
        a = self.clf.classify(text)
        b = self.clf.classify(text)
        self.assertEqual(a["fingerprint"], b["fingerprint"])
        self.assertEqual(a["candidates"], b["candidates"])
        self.assertEqual(a["status"], "confident")
        self.assertEqual(a["label_ids"], ["class_1"])

    def test_boundary_returns_candidates(self):
        result = self.clf.classify("财务审计人员使用机器学习模型检查发票系统")
        self.assertTrue(result["candidates"])
        self.assertEqual(len({c["class_id"] for c in result["candidates"]}), 2)
        self.assertIn(result["status"], {"ambiguous", "multi_label"})

    def test_out_of_domain_review(self):
        result = self.clf.classify("周末去公园跑步，晚上和朋友吃饭看电影")
        self.assertEqual(result["status"], "review")
        self.assertEqual(result["label_ids"], [])

    def test_multi_label_example_can_match_both_classes(self):
        self.clf.add_document(
            "财务技术团队用机器学习模型自动审计发票和报销系统",
            ["class_1", "class_2"])
        result = self.clf.classify("机器学习模型自动审计发票和报销系统")
        self.assertGreaterEqual(len(result["candidates"]), 2)
        self.assertIn(result["status"], {"ambiguous", "multi_label"})

    def test_incremental_boundary_changes_without_rebuild(self):
        text = "财务团队上线自动发票识别模型并维护报销系统"
        before = self.clf.classify(text)
        version = self.clf.model_version
        self.clf.add_document(
            "财务团队开发发票识别模型，把报销审批系统接入自动审计", ["class_1"])
        self.assertEqual(self.clf.model_version, version + 1)
        after = self.clf.classify(text)
        self.assertEqual(after["candidates"][0]["class_id"], "class_1")

    def test_delete_example_updates_stats(self):
        terms = self.clf.tokenize(self.finance[0])
        before = self.clf.class_counts["class_1"]
        self.clf.remove_document(terms, ["class_1"])
        self.assertEqual(self.clf.class_counts["class_1"], before - 1)

    def test_save_load(self):
        path = tempfile.mktemp(suffix=".json")
        try:
            self.clf.save(path)
            restored = IncrementalTextClassifier(path=path)
            text = "算法工程师部署机器学习接口并修复服务器漏洞"
            a = self.clf.classify(text)
            b = restored.classify(text)
            self.assertEqual(a["fingerprint"], b["fingerprint"])
            self.assertEqual(a["candidates"], b["candidates"])
        finally:
            if os.path.exists(path):
                os.remove(path)


@unittest.skipIf(create_app is None, reason="未安装 Flask")
class TestClassifierAPI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.client = create_app(self.tmp).test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _create_examples(self):
        classes = []
        for name, desc in [("财务", "发票预算"), ("技术", "算法系统")]:
            resp = self.client.post("/api/classifier/classes",
                                    json={"name": name, "description": desc})
            classes.append(resp.get_json()["id"])
        examples = [
            (classes[0], "财务审计发票报销预算和税务申报"),
            (classes[0], "公司核算营业收入成本利润和现金流"),
            (classes[1], "算法团队训练机器学习模型并优化服务器"),
            (classes[1], "后端发布微服务接口并修复系统漏洞"),
        ]
        for cid, text in examples:
            resp = self.client.post("/api/classifier/examples",
                                    json={"label_ids": [cid], "text": text})
            self.assertEqual(resp.status_code, 200)
        return classes

    def test_train_classify_and_incremental_persistence(self):
        classes = self._create_examples()
        resp = self.client.post("/api/classifier/classify", json={
            "text": "请提交发票，财务部门完成报销审计"
        })
        self.assertEqual(resp.status_code, 200)
        result = resp.get_json()
        self.assertEqual(result["label_ids"], [classes[0]])
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "models", "classifier.json")))

        new_client = create_app(self.tmp).test_client()
        resp = new_client.get("/api/classifier/model")
        self.assertEqual(resp.get_json()["stats"]["document_count"], 4)
        resp = new_client.post("/api/classifier/classify", json={
            "text": "请提交发票，财务部门完成报销审计"
        })
        self.assertEqual(resp.get_json()["label_ids"], [classes[0]])

    def test_mixed_example_supports_multiple_labels(self):
        classes = self._create_examples()
        resp = self.client.post("/api/classifier/examples", json={
            "label_ids": classes,
            "text": "财务审计模型自动识别发票并接入报销系统",
        })
        self.assertEqual(resp.status_code, 200)
        record = self.client.get("/api/classifier/examples").get_json()["examples"][0]
        self.assertEqual(set(record["label_ids"]), set(classes))


class TestHMM(unittest.TestCase):
    def test_viterbi(self):
        hmm = HMM(["A", "B"], add_k=0.1)
        hmm.train([[(1, "A"), (2, "B")], [(1, "A"), (2, "B")], [(2, "B"), (1, "A")]])
        path = hmm.viterbi([1, 2])
        self.assertEqual(len(path), 2)
        self.assertIn(path[0], ("A", "B"))


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shard_insert_query(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        store.insert_many([{"v": i} for i in range(25)])
        self.assertEqual(store.stats()["total"], 25)
        self.assertEqual(store.stats()["shard_count"], 3)
        self.assertEqual(len(store.query(where=[("v", "gt", 20)])), 4)
        self.assertEqual(len(store.query(where=[("v", "in", [1, 2, 3])])), 3)

    def test_update(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        rid = store.insert({"name": "old", "v": 1})
        updated = store.update(rid, {"name": "new", "v": 2})
        self.assertEqual(updated["name"], "new")
        self.assertEqual(store.get(rid)["v"], 2)

    def test_delete_compact(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        ids = store.insert_many([{"v": i} for i in range(15)])
        store.delete(ids[0])
        stats = store.compact()
        self.assertEqual(stats["records"], 14)

    def test_concurrent_insert(self):
        store = ShardedStore(self.tmp, "t", shard_size=20)
        errors = []

        def worker(offset):
            try:
                store.insert_many([{"v": offset * 1000 + i} for i in range(30)])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)
        self.assertEqual(store.stats()["total"], 180)

    def test_registry_tasks(self):
        reg = StoreRegistry(self.tmp)
        reg.task("a").insert({"x": 1})
        reg.task("b").insert({"x": 2})
        # 造一个非存储目录，不应被识别为任务
        os.makedirs(os.path.join(self.tmp, "models"))
        self.assertEqual(reg.tasks(), ["a", "b"])


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.engine = PipelineEngine().register_builtin()

    def test_run_chain(self):
        cfg = {"name": "p", "stages": [
            {"name": "segment"}, {"name": "pos"}, {"name": "sentiment"}]}
        out = self.engine.build(cfg).run({"text": "这个产品非常好用"})
        self.assertIn("words", out)
        self.assertIn("pos", out)
        self.assertIn("sentiment", out)

    def test_classify_stage(self):
        from nlp import get_classifier
        clf = get_classifier()
        clf.classes = {}
        clf.model_version = 0
        clf.next_class_id = 1
        clf.document_count = 0
        clf.df = {}
        clf.class_counts = {}
        clf.class_term_weights = {}
        clf.add_class("class_1", "财务")
        clf.add_class("class_2", "技术")
        clf.add_document("财务审计发票报销预算和税务申报", ["class_1"])
        clf.add_document("算法团队训练机器学习模型并修复系统漏洞", ["class_2"])
        cfg = {"name": "p", "stages": [{"name": "classify"}]}
        out = self.engine.build(cfg).run({"text": "请提交发票报销并完成财务审计"})
        self.assertIn("classification", out)
        self.assertEqual(out["classification"]["label_ids"], ["class_1"])

    def test_batch(self):
        cfg = {"name": "p", "stages": [{"name": "segment"}, {"name": "keywords"}]}
        results = self.engine.run_batch(
            cfg, ["今天天气很好", "这个产品非常好用"], max_workers=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["ok"] for r in results))

    def test_cycle_detected(self):
        cfg = {"name": "p", "stages": [
            {"name": "segment", "deps": ["pos"]},
            {"name": "pos", "deps": ["segment"]},
        ]}
        with self.assertRaises(PipelineError):
            self.engine.build(cfg)

    def test_missing_stage(self):
        cfg = {"name": "p", "stages": [{"name": "not_exist"}]}
        with self.assertRaises(PipelineError):
            self.engine.build(cfg)


if __name__ == "__main__":
    unittest.main()
