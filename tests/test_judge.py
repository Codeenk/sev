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
