"""TinyStories-Instruct: records, prompts, and rule-based constraint checks.

Each source record holds a story and some of: a summary, required words, features
(Dialogue, BadEnding, Twist, Foreshadowing, MoralValue, Conflict), and a "random
sentence" the story must contain. Fields come in varying order and records are
separated by <|endoftext|> lines.

Three constraints can be checked by rules, and those checks are the reward for
reinforcement learning later: every required word appears (allowing inflections),
the random sentence appears, and the story has quoted speech when Dialogue is
requested. Summary and the other features stay in the prompt but are not scored.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from forge.bpe import ENDOFTEXT

REPOSITORY = "roneneldan/TinyStoriesInstruct"
REVISION = "ee050ed1f8720795be342921335e821856a2b42e"
FIELDS = {
    "Summary:": "summary",
    "Words:": "words",
    "Features:": "features",
    "Random sentence:": "sentence",
    "Story:": "story",
}
PROMPT_ORDER = ("summary", "words", "features", "sentence")
LABELS = {value: key for key, value in FIELDS.items()}


# ---------------------------------------------------------------- parsing


def parse_record(text):
    """One record's text -> dict, or None when it has no story."""
    record = {"summary": None, "words": [], "features": [], "sentence": None, "story": None}
    current, story_lines = None, []
    for line in text.splitlines():
        field = next((key for key in FIELDS if line.startswith(key)), None)
        if field:
            current = FIELDS[field]
            value = line[len(field) :].strip()
            if current == "story":
                story_lines = [value] if value else []
            elif current in ("words", "features"):
                record[current] = [item.strip() for item in value.split(",") if item.strip()]
            else:
                record[current] = value or None
        elif current == "story":
            story_lines.append(line)
    story = "\n".join(story_lines).strip()
    if not story:
        return None
    record["story"] = story
    return record


def iter_records(path):
    """Yield (record or None, raw text) for each <|endoftext|>-separated record."""
    lines = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() == ENDOFTEXT:
                yield _finish(lines)
                lines = []
            else:
                lines.append(line.rstrip("\n"))
    if any(line.strip() for line in lines):
        yield _finish(lines)


def _finish(lines):
    text = "\n".join(lines)
    return parse_record(text), text


def prompt(record):
    """Canonical prompt: present fields in a fixed order, ending where the story starts."""
    lines = []
    for field in PROMPT_ORDER:
        value = record.get(field)
        if not value:
            continue
        if isinstance(value, list):
            value = ", ".join(value)
        lines.append(f"{LABELS[field]} {value}")
    lines.append("Story:\n")
    return "\n".join(lines)


def verifiable(record):
    return bool(record["words"] or record["sentence"] or "Dialogue" in record["features"])


# ---------------------------------------------------------------- checks

IRREGULAR = {
    "be": ["is", "are", "was", "were", "been", "being", "am"],
    "begin": ["began", "begun"],
    "bite": ["bit", "bitten"],
    "blow": ["blew", "blown"],
    "break": ["broke", "broken"],
    "bring": ["brought"],
    "build": ["built"],
    "buy": ["bought"],
    "catch": ["caught"],
    "choose": ["chose", "chosen"],
    "come": ["came"],
    "dig": ["dug"],
    "do": ["did", "done", "does"],
    "draw": ["drew", "drawn"],
    "dream": ["dreamt"],
    "drink": ["drank", "drunk"],
    "drive": ["drove", "driven"],
    "eat": ["ate", "eaten"],
    "fall": ["fell", "fallen"],
    "feed": ["fed"],
    "feel": ["felt"],
    "fight": ["fought"],
    "find": ["found"],
    "fly": ["flew", "flown", "flies"],
    "forget": ["forgot", "forgotten"],
    "freeze": ["froze", "frozen"],
    "get": ["got", "gotten"],
    "give": ["gave", "given"],
    "go": ["went", "gone", "goes"],
    "grow": ["grew", "grown"],
    "hang": ["hung"],
    "have": ["has", "had"],
    "hear": ["heard"],
    "hide": ["hid", "hidden"],
    "hold": ["held"],
    "hurt": ["hurt"],
    "keep": ["kept"],
    "know": ["knew", "known"],
    "lead": ["led"],
    "leave": ["left"],
    "lend": ["lent"],
    "lie": ["lay", "lain", "lying"],
    "light": ["lit"],
    "lose": ["lost"],
    "make": ["made"],
    "mean": ["meant"],
    "meet": ["met"],
    "pay": ["paid"],
    "ride": ["rode", "ridden"],
    "ring": ["rang", "rung"],
    "rise": ["rose", "risen"],
    "run": ["ran"],
    "say": ["said"],
    "see": ["saw", "seen"],
    "seek": ["sought"],
    "sell": ["sold"],
    "send": ["sent"],
    "shake": ["shook", "shaken"],
    "shine": ["shone"],
    "shoot": ["shot"],
    "sing": ["sang", "sung"],
    "sink": ["sank", "sunk"],
    "sit": ["sat"],
    "sleep": ["slept"],
    "slide": ["slid"],
    "speak": ["spoke", "spoken"],
    "spend": ["spent"],
    "spin": ["spun"],
    "stand": ["stood"],
    "steal": ["stole", "stolen"],
    "stick": ["stuck"],
    "sting": ["stung"],
    "strike": ["struck"],
    "swim": ["swam", "swum"],
    "swing": ["swung"],
    "take": ["took", "taken"],
    "teach": ["taught"],
    "tear": ["tore", "torn"],
    "tell": ["told"],
    "think": ["thought"],
    "throw": ["threw", "thrown"],
    "understand": ["understood"],
    "wake": ["woke", "woken"],
    "wear": ["wore", "worn"],
    "weep": ["wept"],
    "win": ["won"],
    "write": ["wrote", "written"],
    "good": ["better", "best"],
    "bad": ["worse", "worst"],
    "child": ["children"],
    "foot": ["feet"],
    "mouse": ["mice"],
    "tooth": ["teeth"],
    "man": ["men"],
    "woman": ["women"],
    "person": ["people"],
    "leaf": ["leaves"],
    "wolf": ["wolves"],
    "knife": ["knives"],
    "shelf": ["shelves"],
    "loaf": ["loaves"],
}
VOWELS = set("aeiou")
WORD = re.compile(r"[a-z]+")


def forms(word):
    """Spellings that count as `word`: itself, regular inflections, irregular forms."""
    w = word.lower()
    result = {w + suffix for suffix in ("", "s", "es", "ed", "d", "ing", "er", "est", "ly", "y")}
    if w.endswith("e"):
        result |= {w[:-1] + "ing", w[:-1] + "er", w[:-1] + "est", w[:-1] + "y"}
    if w.endswith("y") and len(w) > 2 and w[-2] not in VOWELS:
        stem = w[:-1]
        result |= {stem + "ies", stem + "ied", stem + "ier", stem + "iest", stem + "ily"}
    if w.endswith("ie"):
        result.add(w[:-2] + "ying")
    if len(w) >= 3 and w[-1] not in VOWELS | {"w", "x", "y"} and w[-2] in VOWELS:
        if w[-3] not in VOWELS:  # consonant-vowel-consonant: hop -> hopped
            result |= {w + w[-1] + suffix for suffix in ("ed", "ing", "er", "est", "y")}
    if w.endswith("c"):
        result |= {w + "ked", w + "king"}
    if w.endswith("f"):
        result.add(w[:-1] + "ves")
    if w.endswith("fe"):
        result.add(w[:-2] + "ves")
    result |= set(IRREGULAR.get(w, []))
    return result


def normalize(text):
    text = text.lower()
    for fancy, plain in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")):
        text = text.replace(fancy, plain)
    return " ".join(text.split())


# Double-quoted text, or single-quoted speech that opens at a word start and closes
# after punctuation (so apostrophes in "Lily's" or "kids' toys" do not count).
QUOTED = re.compile(r'"[^"]*[a-z][^"]*"|(?:^|\s)\'[a-z][^\'\n]{2,}[,.!?]\'(?:\s|$)')


def has_word(word, text, tokens):
    """`word` or an inflection of it appears as a whole word. Words with
    non-letters (x-ray, ice-cream) are matched in the text itself."""
    if WORD.fullmatch(word.lower()):
        return bool(forms(word) & tokens)
    pattern = r"(?<![a-z])" + re.escape(word.lower()) + r"(?:s|es|ed|d|ing)?(?![a-z])"
    return re.search(pattern, text) is not None


def check(record, story):
    """Score `story` against `record`'s verifiable constraints.

    `reward` averages the applicable constraint types (words counted as the fraction
    found), in [0, 1]; `satisfied` means every applicable check passed. With no
    verifiable constraint, reward is None.
    """
    text = normalize(story)
    tokens = set(WORD.findall(text))
    result = {}
    if record["words"]:
        hits = [has_word(word, text, tokens) for word in record["words"]]
        result["words"] = {"hits": hits, "fraction": sum(hits) / len(hits)}
    if record["sentence"]:
        result["sentence"] = normalize(record["sentence"]).rstrip(".!?") in text
    if "Dialogue" in record["features"]:
        result["dialogue"] = bool(QUOTED.search(text))
    scores = []
    if "words" in result:
        scores.append(result["words"]["fraction"])
    if "sentence" in result:
        scores.append(float(result["sentence"]))
    if "dialogue" in result:
        scores.append(float(result["dialogue"]))
    result["reward"] = sum(scores) / len(scores) if scores else None
    result["satisfied"] = bool(scores) and all(score == 1.0 for score in scores)
    return result


# ---------------------------------------------------------------- data files


def prepare(train, valid, out, eval_size=500, seed=17):
    """Parse both splits to JSONL, draw the fixed evaluation prompts from valid."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": REPOSITORY,
        "revision": REVISION,
        "licenses": ["CDLA-Sharing-1.0"],
        "source_files": [],
        "splits": {},
    }
    valid_records = []
    for split, path in (("train", Path(train)), ("valid", Path(valid))):
        counts = {"records": 0, "skipped_no_story": 0, "verifiable": 0}
        presence = {field: 0 for field in PROMPT_ORDER}
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for index, (record, _) in enumerate(iter_records(path)):
                if record is None:
                    counts["skipped_no_story"] += 1
                    continue
                record = {"id": f"{split}-{index:07d}", **record}
                stream.write(json.dumps(record) + "\n")
                counts["records"] += 1
                counts["verifiable"] += verifiable(record)
                for field in PROMPT_ORDER:
                    presence[field] += bool(record[field])
                if split == "valid":
                    valid_records.append(record)
        counts["field_presence"] = presence
        manifest["splits"][split] = counts
        manifest["source_files"].append(
            {"name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        )
    candidates = [r for r in valid_records if verifiable(r)]
    chosen = sorted(random.Random(seed).sample(range(len(candidates)), eval_size))
    with (out / "eval_prompts.jsonl").open("w", encoding="utf-8") as stream:
        for index in chosen:
            stream.write(json.dumps(candidates[index]) + "\n")
    manifest["eval_prompts"] = {"count": eval_size, "seed": seed, "drawn_from": "valid"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    # utf-8-sig also accepts files that Windows tools saved with a byte-order mark.
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate(records, stories):
    """Constraint satisfaction of generated `stories` ({id: text}) on `records`."""
    totals = {"words": [], "words_all": [], "sentence": [], "dialogue": []}
    rewards, satisfied, missing = [], [], []
    for record in records:
        if record["id"] not in stories:
            missing.append(record["id"])
            continue
        result = check(record, stories[record["id"]])
        if "words" in result:
            totals["words"].extend(result["words"]["hits"])
            totals["words_all"].append(result["words"]["fraction"] == 1.0)
        for key in ("sentence", "dialogue"):
            if key in result:
                totals[key].append(result[key])
        if result["reward"] is not None:
            rewards.append(result["reward"])
            satisfied.append(result["satisfied"])

    def rate(values):
        return {"rate": sum(values) / len(values) if values else None, "n": len(values)}

    return {
        "satisfied": rate(satisfied),
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "per_constraint": {
            "word_recall": rate(totals["words"]),
            "all_words": rate(totals["words_all"]),
            "sentence": rate(totals["sentence"]),
            "dialogue": rate(totals["dialogue"]),
        },
        "scored": len(rewards),
        "missing_ids": missing,
    }


def gold_check(records, seed=17):
    """How the checks score the dataset's own stories, and a shuffled base rate:
    the same stories scored against a different record's constraints."""
    own = evaluate(records, {r["id"]: r["story"] for r in records})
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    shuffled = {records[i]["id"]: records[j]["story"] for i, j in zip(range(len(records)), order)}
    failures = []
    for record in records:
        result = check(record, record["story"])
        if result["reward"] is not None and not result["satisfied"]:
            failures.append(
                {
                    "id": record["id"],
                    "result": result,
                    "words": record["words"],
                    "sentence": record["sentence"],
                    "features": record["features"],
                }
            )
    return {"own_story": own, "shuffled_story": evaluate(records, shuffled), "failures": failures}
