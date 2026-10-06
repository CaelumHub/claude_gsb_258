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
from nlp.classifier import DocumentClassifier
from nlp.hmm import HMM
from pipeline import PipelineEngine, PipelineError
from storage import ShardedStore, StoreRegistry


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


class TestClassifier(unittest.TestCase):
    """文档自动分类：少样本学习、增量一致性、确定性、拒识与多候选。"""

    EXAMPLES = {
        "cat_sports": ("体育", ["足球比赛球队进球获胜", "篮球联赛球员夺得冠军",
                              "运动员在奥运会比赛中获得金牌"]),
        "cat_tech": ("科技", ["计算机软件算法与人工智能", "互联网平台数据系统开发",
                            "机器学习模型深度神经网络"]),
        "cat_finance": ("财经", ["股票市场投资与经济增长", "银行贷款利率金融市场",
                              "企业利润营收财报"]),
    }

    def _trained(self) -> DocumentClassifier:
        clf = DocumentClassifier()
        for cid, (name, texts) in self.EXAMPLES.items():
            clf.add_category(cid, name)
            for t in texts:
                clf.add_example(cid, t)
        return clf

    def test_basic_label_and_confidence(self):
        clf = self._trained()
        r = clf.classify("足球运动员在决赛中进球")
        self.assertTrue(r["accepted"])
        self.assertEqual(r["label"], "体育")
        self.assertGreater(r["confidence"], 0)
        self.assertLessEqual(r["confidence"], 1)
        self.assertTrue(r["candidates"])
        self.assertIn("matched", r["candidates"][0])

    def test_deterministic_repeated_classify(self):
        clf = self._trained()
        r1 = clf.classify("互联网平台使用机器学习算法")
        r2 = clf.classify("互联网平台使用机器学习算法")
        self.assertEqual(r1, r2)

    def test_incremental_equals_full_rebuild(self):
        """增量学习的状态必须与全量重建逐位一致（边界随示例调整）。"""
        clf = self._trained()
        rebuilt = DocumentClassifier()
        rebuilt.rebuild([(cid, name, t)
                         for cid, (name, texts) in self.EXAMPLES.items()
                         for t in texts])
        self.assertEqual(clf.fingerprint(), rebuilt.fingerprint())
        self.assertEqual(clf.classify("股票市场上涨"),
                         rebuilt.classify("股票市场上涨"))

    def test_remove_example_exact_undo(self):
        """删除示例精确抵消其贡献，模型回到添加前的状态。"""
        clf = self._trained()
        before = clf.fingerprint()
        clf.add_example("cat_sports", "排球锦标赛夺得冠军")
        self.assertNotEqual(clf.fingerprint(), before)
        clf.remove_example("cat_sports", "排球锦标赛夺得冠军")
        self.assertEqual(clf.fingerprint(), before)

    def test_ambiguous_returns_multiple_candidates(self):
        """边界模糊的文档应给出多个候选，而不是硬塞一个类别。"""
        clf = DocumentClassifier()
        clf.add_category("a", "甲类")
        clf.add_example("a", "apple banana")
        clf.add_category("b", "乙类")
        clf.add_example("b", "apple cherry")
        r = clf.classify("apple")
        self.assertTrue(r["accepted"])
        self.assertTrue(r["ambiguous"])
        self.assertEqual(len(r["candidates"]), 2)
        self.assertEqual(r["candidates"][0]["confidence"],
                         r["candidates"][1]["confidence"])

    def test_abstain_on_unrelated_text(self):
        """与所有类别都无关的文档应拒识，不乱塞类别。"""
        clf = self._trained()
        r = clf.classify("芭蕾歌剧")
        self.assertFalse(r["accepted"])
        self.assertIsNone(r["label"])
        self.assertEqual(r["candidates"], [])

    def test_save_load_roundtrip(self):
        clf = self._trained()
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "classifier.json")
            clf.save(path)
            restored = DocumentClassifier()
            restored.load(path)
            self.assertEqual(clf.fingerprint(), restored.fingerprint())
            self.assertEqual(clf.classify("篮球比赛"),
                             restored.classify("篮球比赛"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_untrained_model_raises(self):
        with self.assertRaises(ValueError):
            DocumentClassifier().classify("任意文本")

    def test_empty_example_rejected(self):
        clf = DocumentClassifier()
        with self.assertRaises(ValueError):
            clf.add_example("c1", "！！！")

    def test_remove_category_updates_stats(self):
        clf = self._trained()
        clf.remove_category("cat_finance")
        self.assertEqual(clf.stats()["n_docs"], 6)
        r = clf.classify("股票市场投资")
        self.assertNotIn("财经", [c["name"] for c in r["candidates"]])


if __name__ == "__main__":
    unittest.main()
