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


def test_training_context_keys_are_encode_compatible():
    """training_context() adds max_packed, which fits() accepts but encode() does not: any code that splats the
    context into encode() raises TypeError at runtime. This is the guard for that whole bug class."""
    import inspect
    from kev.model import MAX_TRAIN_STATE, encode, training_context
    params = set(inspect.signature(encode).parameters)
    for ms in (384, 2048, MAX_TRAIN_STATE):
        extra = set(training_context(ms)) - params
        assert extra <= {"max_packed"}, (ms, extra)
    # and the two limits encode does take must actually be present, or the call is silently wrong
    assert {"max_state", "max_branch"} <= set(training_context(2048))


def _stub_datasets():
    """kev.data imports `datasets`, which the weight-free unit job does not install. Stubbing the module
    lets the REAL training_requests path run here; nothing in this test touches the Hub."""
    import sys, types
    if "datasets" not in sys.modules:
        m = types.ModuleType("datasets")
        m.load_dataset = lambda *a, **k: None
        sys.modules["datasets"] = m


def test_smoke_worst_survives_real_record_shapes(monkeypatch):
    """--smoke_worst runs on real records, so it must survive the real record shape. Two bugs lived here:
    training_context() carries max_packed, which encode() rejects (TypeError); and the option count must come
    from materialize()'s LIST of questions, not the raw dict and not enc["judge"] (which only DecisionModel
    builds). Both crashed the gate on its first rung."""
    _stub_datasets()
    import argparse as ap
    from kev import train as T
    from kev.model import load_tokenizer

    def rec(n_opts, state_words):
        return {"state": " ".join(["w"] * state_words),
                "_meta": {"id": f"i{n_opts}_{state_words}", "source": "unit"},
                "questions": {"q0": {"type": "choice", "instructions": "pick one",
                                      "criteria": {f"opt{k}": None for k in range(n_opts)},
                                      "label": "opt0", "src": "unit"}}}

    # option-heavy and state-heavy records, plus a cheap one that must NOT be selected
    pool = [rec(2, 10), rec(9, 10), rec(2, 400), rec(3, 10)]
    monkeypatch.setattr(T, "load_records", lambda p: pool)
    a = ap.Namespace(data="x", suite="", replay=0, max_state=2048, seed=0, n_per_source=1000,
                     train_sources="", public_frac=1.0, synthetic_repeat=1, holdout=0, smoke_worst=2)
    kept = T.training_requests(a, load_tokenizer("Qwen/Qwen3-0.6B-Base"), None, 0)
    assert len(kept) == 2
    picked = {r["_meta"]["id"] for r in kept}
    # the 9-option record costs ~5x the 2-option one, so it must beat the 400-word state record
    assert "i9_10" in picked, picked


def test_judge_option_cap_keeps_correct_and_remaps():
    """Judge training replicates the state KV per option row, so a 77-option question costs 78 rows and OOMs
    at any max_state. The cap keeps correct + sampled distractors with remapped labels/keys (eval still scores
    the full set), deterministically in the rng; the backstop never drops a correct option or a question."""
    _stub_datasets()
    import random
    from kev.train import subsample_judge_options

    def Q(n, lab=0):
        return {"instr": "pick", "options": [f"opt{i}" for i in range(n)], "label": lab,
                "keys": [f"k{i}" for i in range(n)], "qid": "q", "qtype": "choice", "src": "u"}

    narrow = subsample_judge_options({"state": "s", "questions": [Q(5, 2)]}, random.Random(0))
    assert narrow["questions"][0]["label"] == 2 and len(narrow["questions"][0]["options"]) == 5
    a = subsample_judge_options({"state": "s", "questions": [Q(77, 41)]}, random.Random(7))
    b = subsample_judge_options({"state": "s", "questions": [Q(77, 41)]}, random.Random(7))
    qa = a["questions"][0]
    assert len(qa["options"]) == 8 and qa["options"][qa["label"]] == "opt41" and a == b
    for o, k in zip(qa["options"], qa["keys"]):
        assert k == "k" + o[3:]
    rec = {"state": " ".join(["w"] * 200), "questions": [Q(77, i) for i in range(6)]}
    c = subsample_judge_options(rec, random.Random(3), keep=8, rowtoken_budget=4000)
    for i, q in enumerate(c["questions"]):
        assert q["options"][q["label"]] == f"opt{i}" and len(q["options"]) >= 2
