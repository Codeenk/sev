"""Sev-X judge readout: weight-free unit tests (template, token gates, payload shape)."""
import pytest


def test_judge_row_text_carries_all_roles():
    from kev.model import judge_row_text, JUDGE_INSTRUCT
    s = judge_row_text("Which team?", "returns", "broken items")
    assert JUDGE_INSTRUCT in s and "Which team?" in s and "returns" in s and "broken items" in s


def test_judge_template_is_cheap():
    from transformers import AutoTokenizer
    from kev.model import JUDGE_PREFIX, JUDGE_SUFFIX
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    assert len(tok.encode(JUDGE_PREFIX, add_special_tokens=False)) < 64
    assert len(tok.encode(JUDGE_SUFFIX, add_special_tokens=False)) < 16


def test_yes_no_are_single_tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    assert len(tok.encode("yes", add_special_tokens=False)) == 1
    assert len(tok.encode("no", add_special_tokens=False)) == 1


def test_readout_flag_validated_before_download():
    from kev.model import DecisionModel
    import inspect
    src = inspect.getsource(DecisionModel.__init__)
    assert src.index('readout not in ("pointer", "judge")') < src.index("from_pretrained")
    with pytest.raises(ValueError, match="readout must be pointer or judge"):
        DecisionModel("Qwen/Qwen3-0.6B-Base", None, "cpu", readout="bogus")


def test_smoke_worst_selects_heaviest_records(monkeypatch):
    """--smoke_worst must keep the LONGEST records: both v7 OOMs died on the first long record, and a default
    smoke samples the first (short) ones, so it would certify a config that still dies in the real run."""
    import argparse as ap
    train_mod = pytest.importorskip("kev.train", reason="kev.train needs the dataset deps the unit job omits")
    T = train_mod
    recs = [{"_meta": {"id": str(i), "source": "synthetic"}, "state": "word " * (10 * (i + 1)),
             "questions": [{"instr": "pick", "options": ["a", "b"], "label": 0, "keys": ["a", "b"]}]}
            for i in range(6)]
    monkeypatch.setattr(T, "load_records", lambda p: recs)
    a = ap.Namespace(data="x", suite="", replay=0, max_state=4096, seed=0, n_per_source=1000,
                     train_sources="", public_frac=1.0, synthetic_repeat=1, holdout=0, smoke_worst=2)
    kept = T.training_requests(a, load_tokenizer("Qwen/Qwen3-0.6B-Base"), None, 0)
    assert len(kept) == 2
    lens = [len(r["state"]) for r in kept]
    assert lens == sorted(lens, reverse=True), lens
