import baseline as bl

ERRORS = [{"word_index": 1, "type": "fr_homophone", "original": "a", "corrupted": "à"}]
SOURCE = "Il à mangé une pomme."
TARGET = "Il a mangé une pomme."


def counts(output, source=SOURCE, target=TARGET, errors=ERRORS):
    result = bl.classify(source, target, output, errors)
    return {k: len(v) for k, v in result.items()}


def test_fixed():
    assert counts(TARGET) == {"fixed": 1, "missed": 0, "false_positives": 0, "rewrites": 0}


def test_missed_when_left_alone_or_changed_wrongly():
    assert counts(SOURCE)["missed"] == 1
    assert counts("Il as mangé une pomme.") == {"fixed": 0, "missed": 1, "false_positives": 0, "rewrites": 0}


def test_false_positive():
    assert counts("Il a mangé une poire.") == {"fixed": 1, "missed": 0, "false_positives": 1, "rewrites": 0}
    assert counts("il a mangé une pomme.")["false_positives"] == 1  # capitalisation counts


def test_rewrite_on_insertion_and_deletion():
    assert counts("Il a mangé une belle pomme.") == {"fixed": 1, "missed": 0, "false_positives": 0, "rewrites": 1}
    assert counts("Il a mangé.")["rewrites"] == 1


def test_error_inside_a_rewritten_block():
    assert counts("Il a déjà mangé une pomme.") == {"fixed": 1, "missed": 0, "false_positives": 0, "rewrites": 1}
    assert counts("Il avait déjà mangé une pomme.") == {"fixed": 0, "missed": 1, "false_positives": 0, "rewrites": 1}


def test_clean_sentence_untouched():
    assert counts(TARGET, source=TARGET, errors=[]) == {"fixed": 0, "missed": 0, "false_positives": 0, "rewrites": 0}


def test_strip_thinking():
    assert bl.strip_thinking("<think>hmm\nok</think>\nBonjour.") == ("Bonjour.", True)
    assert bl.strip_thinking("hmm</think>Bonjour.") == ("Bonjour.", True)
    assert bl.strip_thinking("<think>never closed") == ("", True)
    assert bl.strip_thinking(" Bonjour. ") == ("Bonjour.", False)


def test_load_prompt(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("ignored\n### system\nBe strict.\n\n### user\nteh\n### assistant\nthe\n", encoding="utf-8")
    assert bl.load_prompt(path) == [
        {"role": "system", "content": "Be strict."},
        {"role": "user", "content": "teh"},
        {"role": "assistant", "content": "the"},
    ]


def test_percentile():
    assert bl.percentile([], 50) is None
    assert bl.percentile([3.0], 95) == 3.0
    assert bl.percentile([1, 2, 3, 4], 50) == 2
    assert bl.percentile(list(range(1, 101)), 95) == 95


def test_summarize_per_error_type():
    row = {
        "lang": "fr", "clean": False, "input": SOURCE, "target": TARGET, "output": TARGET,
        "exact": True, "words": 5, "errors": ERRORS, "latency_s": 1.0, "completion_tokens": 10,
        **bl.classify(SOURCE, TARGET, TARGET, ERRORS),
    }
    clean = {
        **row, "clean": True, "input": TARGET, "output": "Il a mangé une poire.", "exact": False,
        "errors": [], **bl.classify(TARGET, TARGET, "Il a mangé une poire.", []),
    }
    metrics = bl.compute_metrics([row, clean])
    assert metrics["overall"]["recall"] == 1.0
    assert metrics["overall"]["exact_match_rate"] == 0.5
    assert metrics["overall"]["false_positives"] == 1
    assert metrics["overall"]["false_positives_per_100_words"] == 10.0
    assert metrics["overall"]["clean_untouched_rate"] == 0.0
    assert metrics["overall"]["tokens_per_sec"] == 10.0
    assert metrics["by_error_type"]["fr_homophone"]["examples"] == 1
    assert metrics["by_error_type"]["fr_homophone"]["false_positives"] == 0
