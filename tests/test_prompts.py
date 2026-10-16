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
