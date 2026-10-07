import json

import pytest

from forge.instruct import (
    check,
    evaluate,
    forms,
    iter_records,
    parse_record,
    prepare,
    prompt,
    repetition,
)

SOURCE = """Features: Dialogue, BadEnding
Summary: Ben and Mia get lost.
Random sentence: They wished they had never been eager.
Story:
Ben and Mia were eager to ride the subway.
"Stop!" said Mom. They wished they had never been eager.
<|endoftext|>
Words: escape, war, tall
Story:
Tom and Lily wanted to escape the war. The trees were taller than houses.
Summary: Tom and Lily escape.
<|endoftext|>
Summary: No story here.
<|endoftext|>
"""


@pytest.fixture
def records(tmp_path):
    path = tmp_path / "valid.txt"
    path.write_text(SOURCE, "utf-8")
    return [record for record, _ in iter_records(path)]


def test_parses_fields_in_any_order_including_summary_after_story(records):
    first, second, third = records
    assert first["features"] == ["Dialogue", "BadEnding"]
    assert first["sentence"] == "They wished they had never been eager."
    assert first["story"].startswith("Ben and Mia") and first["story"].endswith("been eager.")
    assert second["words"] == ["escape", "war", "tall"]
    assert second["summary"] == "Tom and Lily escape."
    assert "Summary" not in second["story"]
    assert third is None  # no story


def test_prompt_has_a_fixed_field_order_and_ends_where_the_story_starts(records):
    text = prompt(records[1])
    assert text == "Summary: Tom and Lily escape.\nWords: escape, war, tall\nStory:\n"


def test_gold_stories_satisfy_their_own_constraints(records):
    for record in records[:2]:
        assert check(record, record["story"])["satisfied"]


@pytest.mark.parametrize(
    ("word", "spelling"),
    [
        ("jump", "jumped"),
        ("hop", "hopping"),
        ("happy", "happily"),
        ("try", "tried"),
        ("big", "bigger"),
        ("run", "ran"),
        ("leaf", "leaves"),
        ("understand", "understood"),
        ("sleep", "sleepy"),
    ],
)
def test_inflections_count(word, spelling):
    assert spelling in forms(word)


def test_checks_are_strict_about_missing_words_derivations_and_apostrophes():
    record = {"words": ["value", "x-ray"], "sentence": None, "features": ["Dialogue"]}
    result = check(record, "It was a valuable X-ray. Lily's kids' toys were there.")
    assert result["words"]["hits"] == [False, True]  # "valuable" is not "value"
    assert result["dialogue"] is False  # apostrophes are not speech
    assert result["reward"] == pytest.approx(0.25)
    assert not result["satisfied"]
    spoken = check(record, "The value of an x-ray? 'Look at this,' said Tom.")
    assert spoken["satisfied"]


def test_sentence_matching_ignores_case_quotes_and_spacing():
    record = {"words": [], "sentence": "She said “Hi” to Ben.", "features": []}
    assert check(record, 'Then  she SAID "hi" to ben!')["satisfied"]
    assert not check(record, "She waved at Ben.")["satisfied"]


def test_repetition_counts_repeated_word_four_grams():
    assert repetition("Tom ran to the park and then went home.") == 0.0
    # 5 four-grams (abcd bcda cdab dabc abcd), one a repeat.
    assert repetition("A b c d. A b c d!") == pytest.approx(0.2)
    assert repetition("Too short.") == 0.0


def test_prepare_and_evaluate(tmp_path):
    (tmp_path / "train.txt").write_text(SOURCE, "utf-8")
    (tmp_path / "valid.txt").write_text(SOURCE, "utf-8")
    manifest = prepare(tmp_path / "train.txt", tmp_path / "valid.txt", tmp_path / "out", 2)
    assert manifest["splits"]["valid"]["records"] == 2
    assert manifest["splits"]["valid"]["skipped_no_story"] == 1
    lines = (tmp_path / "out" / "eval_prompts.jsonl").read_text("utf-8").splitlines()
    eval_records = [json.loads(line) for line in lines]
    assert len(eval_records) == 2
    report = evaluate(eval_records, {eval_records[0]["id"]: eval_records[0]["story"]})
    assert report["satisfied"]["rate"] == 1.0
    assert report["missing_ids"] == [eval_records[1]["id"]]
    assert parse_record("Story:\nOnce.")["story"] == "Once."
