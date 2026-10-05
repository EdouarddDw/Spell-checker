import random

import pytest

import make_eval as me

SEEDS = range(50)


def rng(seed=0):
    return random.Random(seed)


@pytest.fixture(scope="module")
def examples():
    return me.build_examples(me.PLACEHOLDERS, seed=42, variants=5)


# ---- tokens --------------------------------------------------------------


@pytest.mark.parametrize(
    "token, expected",
    [
        ("table.", ("", "table", ".")),
        ("«école»,", ("«", "école", "»,")),
        ("(l'été)", ("(", "l'été", ")")),
        ("garde-moi", ("", "garde-moi", "")),
        ("?", ("?", "", "")),
        ("2024", ("2024", "", "")),
    ],
)
def test_split_token(token, expected):
    assert me.split_token(token) == expected


def test_is_word_rejects_digits_and_urls():
    assert me.is_word("l’école")
    assert not me.is_word("v2x")
    assert not me.is_word("a.b")
    assert not me.is_word("")


# ---- keyboard ------------------------------------------------------------


def test_key_neighbours_per_layout():
    assert set(me.key_neighbours("qwerty")["s"]) == set("awedxz")
    assert set(me.key_neighbours("qwertz-ch")["t"]) == set("rzfg")
    assert set(me.key_neighbours("azerty")["a"]) == set("zq")


@pytest.mark.parametrize("layout", sorted(me.KEYBOARD_ROWS))
def test_adjacent_key_uses_a_neighbour(layout):
    neighbours = me.key_neighbours(layout)
    for seed in SEEDS:
        out = me.adjacent_key("Maison", rng(seed), neighbours)
        diffs = [(a, b) for a, b in zip("Maison", out) if a != b]
        assert len(out) == 6 and len(diffs) == 1
        assert diffs[0][1] in neighbours[diffs[0][0]]
        assert out[0] == "M"


# ---- shared typos --------------------------------------------------------


def test_transposition_swaps_two_letters():
    for seed in SEEDS:
        out = me.transposition("fenêtre", rng(seed))
        assert out != "fenêtre" and sorted(out) == sorted("fenêtre") and out[0] == "f"


def test_deletion_and_doubling_change_length_by_one():
    for seed in SEEDS:
        assert len(me.deletion("élève", rng(seed))) == 4
        assert len(me.doubled_letter("élève", rng(seed))) == 6


def test_typos_never_touch_apostrophes_or_first_letter():
    for seed in SEEDS:
        for fn in (me.transposition, me.deletion, me.doubled_letter):
            out = fn("l'école", rng(seed))
            assert out.startswith("l'") and out.count("'") == 1


def test_deletion_never_leaves_a_dangling_apostrophe():
    assert {me.deletion("can't", rng(s)) for s in SEEDS} == {"cn't", "ca't"}


def test_typos_skip_short_words():
    for fn in (me.transposition, me.deletion, me.doubled_letter):
        assert fn("à", rng()) is None
        assert fn("et", rng()) is None
    assert me.adjacent_key("a", rng(), me.key_neighbours("qwerty")) is None
    assert me.transposition("aaa", rng()) is None


# ---- English -------------------------------------------------------------


def test_en_confusion():
    assert me.en_confusion("then", rng()) == "than"
    assert me.en_confusion("Its", rng()) == "It's"
    assert me.en_confusion("it’s", rng()) == "its"
    assert {me.en_confusion("their", rng(s)) for s in SEEDS} == {"there", "they're"}
    assert me.en_confusion("house", rng()) is None


# ---- French --------------------------------------------------------------


def test_accent_strip():
    assert me.accent_strip("élève", rng()) == "eleve"
    assert me.accent_strip("À", rng()) == "A"
    assert me.accent_strip("garçon", rng()) == "garcon"
    assert me.accent_strip("maison", rng()) is None


def test_accent_swap():
    assert {me.accent_swap("thé", rng(s)) for s in SEEDS} == {"thè", "thê"}
    assert me.accent_swap("État", rng()) in {"Ètat", "Êtat"}
    assert me.accent_swap("où", rng()) is None


def test_fr_homophone():
    assert me.fr_homophone("à", rng()) == "a"
    assert me.fr_homophone("Ou", rng()) == "Où"
    assert me.fr_homophone("sont", rng()) == "son"
    assert {me.fr_homophone("c'est", rng(s)) for s in SEEDS} == {"ces", "ses", "s'est"}
    assert me.fr_homophone("C’est", rng(1)) in {"Ces", "Ses", "S’est"}
    assert me.fr_homophone("on", rng(), skip=me.AMBIGUOUS_IN_MIXED) is None
    assert me.fr_homophone("maison", rng()) is None


def test_verb_ending():
    assert {me.verb_ending("manger", rng(s)) for s in SEEDS} == {"mangé", "mangez"}
    assert {me.verb_ending("oublié", rng(s)) for s in SEEDS} == {"oublier", "oubliez"}
    assert me.verb_ending("chez", rng()) is None
    assert me.verb_ending("thé", rng()) is None
    assert me.verb_ending("maison", rng()) is None


def test_dropped_plural():
    assert me.dropped_plural("livres", rng()) == "livre"
    assert me.dropped_plural("très", rng()) is None
    assert me.dropped_plural("les", rng()) is None
    assert me.dropped_plural("Pouvez-vous", rng()) is None
    assert me.dropped_plural("maison", rng()) is None


# ---- whole examples ------------------------------------------------------


def test_deterministic_given_the_seed():
    a = me.build_examples(me.PLACEHOLDERS, seed=7)
    b = me.build_examples(me.PLACEHOLDERS, seed=7)
    c = me.build_examples(me.PLACEHOLDERS, seed=8)
    assert a == b
    assert a != c


def test_clean_examples_are_unchanged(examples):
    clean = [e for e in examples if e["clean"]]
    assert clean
    for e in clean:
        assert e["input"] == e["target"] and e["errors"] == []


def test_clean_ratio(examples):
    for lang in me.LANGS:
        rows = [e for e in examples if e["lang"] == lang]
        assert sum(e["clean"] for e in rows) == round(len(rows) * 0.3)
    assert all(e["clean"] for e in me.build_examples(me.PLACEHOLDERS, clean_ratio=1.0))


def test_non_clean_examples_have_at_least_one_change(examples):
    for e in (e for e in examples if not e["clean"]):
        assert e["input"] != e["target"]
        assert len(e["errors"]) >= 1


def test_errors_describe_exactly_the_changed_words(examples):
    for e in examples:
        src, tgt = e["input"].split(), e["target"].split()
        assert len(src) == len(tgt)
        changed = [i for i, (a, b) in enumerate(zip(src, tgt)) if a != b]
        assert changed == [err["word_index"] for err in e["errors"]]
        for err in e["errors"]:
            assert err["original"] in tgt[err["word_index"]]
            assert err["corrupted"] in src[err["word_index"]]
            assert err["original"] != err["corrupted"]


def test_punctuation_and_capitalisation_preserved(examples):
    def shape(sentence):
        return [(p, s) for p, _, s in map(me.split_token, sentence.split())]

    for e in examples:
        assert shape(e["input"]) == shape(e["target"])
        for a, b in zip(e["input"].split(), e["target"].split()):
            core_a, core_b = me.split_token(a)[1], me.split_token(b)[1]
            if core_a:
                assert core_a[0].isupper() == core_b[0].isupper()


def test_every_error_type_is_produced(examples):
    seen = {(e["lang"], err["type"]) for e in examples for err in e["errors"]}
    for lang in me.LANGS:
        for name in me.corruptions_for(lang, "qwertz-ch"):
            assert (lang, name) in seen


def test_mixed_sentences_get_no_verb_ending_swaps(examples):
    types = {err["type"] for e in examples if e["lang"] == "mixed" for err in e["errors"]}
    assert "verb_ending" not in types


# ---- files ---------------------------------------------------------------


def test_placeholders_created_and_marked(tmp_path):
    created = me.ensure_sentence_files(tmp_path)
    assert len(created) == 3
    for path in created:
        assert me.is_placeholder(path)
        assert len(me.load_sentences(path)) == 20
    assert me.ensure_sentence_files(tmp_path) == []


def test_main_is_reproducible(tmp_path):
    me.main(["--data-dir", str(tmp_path)])
    first = (tmp_path / "eval.jsonl").read_text(encoding="utf-8")
    me.main(["--data-dir", str(tmp_path)])
    assert (tmp_path / "eval.jsonl").read_text(encoding="utf-8") == first
    assert len(first.splitlines()) == 60
