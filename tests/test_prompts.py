import pytest

from train.prompts import STYLES, build_messages


@pytest.mark.parametrize("style", STYLES)
def test_messages_are_well_formed(style):
    msgs = build_messages(style, "statins and breast cancer")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "statins and breast cancer" in msgs[1]["content"]
    assert "<query>" in msgs[0]["content"] and "</query>" in msgs[0]["content"]


def test_styles_differ_and_unknown_style_rejected():
    assert len({build_messages(s, "x")[0]["content"] for s in STYLES}) == len(STYLES)
    with pytest.raises(ValueError):
        build_messages("nope", "x")


def test_fewshot_messages_have_examples_then_query():
    from rewards.reward import parse_expansion
    from train.prompts import FEWSHOT_EXAMPLES, FEWSHOT_STYLE

    msgs = build_messages(FEWSHOT_STYLE, "statins and breast cancer")
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert msgs[0]["content"] == build_messages("keywords", "x")[0]["content"]
    assert msgs[-1]["content"] == "Query: statins and breast cancer"
    for (q, e), user, asst in zip(FEWSHOT_EXAMPLES, msgs[1::2], msgs[2::2]):
        assert user["content"] == f"Query: {q}" and parse_expansion(asst["content"]) == e


def test_day2_styles_unchanged_by_fewshot():
    from train.prompts import FEWSHOT_STYLE
    assert FEWSHOT_STYLE not in STYLES and len(build_messages("keywords", "x")) == 2


def test_fingerprint_covers_fewshot_examples(monkeypatch):
    from train import prompts
    base = prompts.prompt_fingerprint("keywords_fs")
    assert base != prompts.prompt_fingerprint("keywords")  # same system prompt, different examples
    monkeypatch.setattr(prompts, "FEWSHOT_EXAMPLES", prompts.FEWSHOT_EXAMPLES[:2])
    assert prompts.prompt_fingerprint("keywords_fs") != base
