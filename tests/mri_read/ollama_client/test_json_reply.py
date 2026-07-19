import pytest

from mri_read.ollama_client.json_reply import parse_json_reply


def test_parses_plain_json():
    assert parse_json_reply('{"a": 1}') == {"a": 1}


def test_strips_markdown_json_fence():
    text = '```json\n{"a": 1}\n```'
    assert parse_json_reply(text) == {"a": 1}


def test_strips_plain_fence_without_json_hint():
    text = '```\n{"a": 1}\n```'
    assert parse_json_reply(text) == {"a": 1}


def test_extracts_json_from_surrounding_prose():
    text = 'Sure, here you go:\n{"a": 1, "b": [1, 2]}\nHope that helps!'
    assert parse_json_reply(text) == {"a": 1, "b": [1, 2]}


def test_no_braces_raises():
    with pytest.raises(ValueError):
        parse_json_reply("no json here")


# --- adversarial / malformed LLM replies -----------------------------------
# parse_json_reply's callers (ollama_vision.engine_impl, agent.synthesis) all
# catch only `(ValueError, IndexError)` around this call and fall back to an
# "unparsed" result rather than crashing -- these tests pin that every
# malformed-input path raises one of those two types, not something else
# that would slip past that narrow except and propagate uncaught.

def test_empty_string_raises_value_error():
    with pytest.raises(ValueError):
        parse_json_reply("")


def test_reversed_braces_raises_value_error():
    with pytest.raises(ValueError):
        parse_json_reply("} some text {")


def test_empty_object_parses_to_empty_dict():
    assert parse_json_reply("{}") == {}


def test_multiple_json_objects_in_reply_raises_value_error_not_crash():
    # first-{ to last-} spans across both objects and the text between them
    # -- not valid JSON, but must fail as ValueError, not something uncaught.
    text = '{"a": 1} here is another example: {"b": 2}'
    with pytest.raises(ValueError):
        parse_json_reply(text)


def test_unicode_content_round_trips():
    text = '{"impression": "no findings — normal café/böyle"}'
    assert parse_json_reply(text)["impression"] == "no findings — normal café/böyle"


def test_nan_literal_is_tolerated_not_crashed_on():
    """Python's json module accepts the non-standard NaN/Infinity tokens by
    default -- an LLM asked to fill in a numeric field could plausibly emit
    one. This must not crash; the resulting float('nan') is the caller's
    concern (e.g. when re-serializing into report.json), not this parser's.
    """
    import math
    result = parse_json_reply('{"confidence_score": NaN}')
    assert math.isnan(result["confidence_score"])
