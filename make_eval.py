#!/usr/bin/env python3
"""Build data/eval.jsonl by injecting spelling errors into clean sentences.

Reads data/sentences_{en,fr,mixed}.txt (one clean sentence per line), leaves a
share of them untouched, corrupts the rest, and writes one JSON object per line:

    {id, lang, input, target, clean, errors: [{word_index, type, original, corrupted}]}

`word_index` is the index of the whitespace-separated token. Corruptions never
add or remove tokens, so input and target always have the same token count.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from functools import partial
from pathlib import Path

LANGS = ("en", "fr", "mixed")
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
PLACEHOLDER_MARK = "# PLACEHOLDER"

# --------------------------------------------------------------------------
# Keyboard layouts
# --------------------------------------------------------------------------

# Letter rows only. qwertz-ch is the Swiss French variant: è sits right of P,
# é and à right of L.
KEYBOARD_ROWS = {
    "qwerty": ("qwertyuiop", "asdfghjkl", "zxcvbnm"),
    "azerty": ("azertyuiop", "qsdfghjklm", "wxcvbn"),
    "qwertz-ch": ("qwertzuiopè", "asdfghjkléà", "yxcvbnm"),
}
ROW_OFFSETS = (0.0, 0.25, 0.75)  # physical stagger of the three rows, in key widths


def key_neighbours(layout: str) -> dict[str, str]:
    """Map each key to the keys physically touching it."""
    rows = KEYBOARD_ROWS[layout]
    pos = {
        ch: (r, c + ROW_OFFSETS[r])
        for r, row in enumerate(rows)
        for c, ch in enumerate(row)
    }
    return {
        ch: "".join(
            other
            for other, (r2, x2) in pos.items()
            if other != ch and abs(r2 - r) <= 1 and abs(x2 - x) <= 1
        )
        for ch, (r, x) in pos.items()
    }


# --------------------------------------------------------------------------
# Word helpers
# --------------------------------------------------------------------------

APOSTROPHES = "'’"
MIN_TYPO_LETTERS = 3


def split_token(token: str) -> tuple[str, str, str]:
    """Split a token into (leading punctuation, word core, trailing punctuation)."""
    letters = [i for i, ch in enumerate(token) if ch.isalpha()]
    if not letters:
        return token, "", ""
    first, last = letters[0], letters[-1]
    return token[:first], token[first : last + 1], token[last + 1 :]


def is_word(core: str) -> bool:
    """True for cores made of letters, apostrophes and hyphens only (no digits, URLs...)."""
    return bool(core) and all(ch.isalpha() or ch in APOSTROPHES + "-" for ch in core)


def match_case(original: str, replacement: str) -> str:
    """Give `replacement` the capitalisation of `original`."""
    if len(original) > 1 and original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _typo_positions(word: str) -> list[int]:
    """Letter positions a typo may hit. The first letter is spared so that
    capitalisation is never affected."""
    if sum(ch.isalpha() for ch in word) < MIN_TYPO_LETTERS:
        return []
    return [i for i in range(1, len(word)) if word[i].isalpha()]


def _swap_within_groups(word, rng, groups, skip=()):
    """Replace `word` by another member of its confusion group, if it has one."""
    key = word.lower().replace("’", "'")
    if key in skip:
        return None
    for group in groups:
        if key in group:
            new = rng.choice([w for w in group if w != key])
            if "’" in word:
                new = new.replace("'", "’")
            return match_case(word, new)
    return None


# --------------------------------------------------------------------------
# Shared corruptions (typos). Each takes a word core and returns the corrupted
# word, or None when it does not apply.
# --------------------------------------------------------------------------


def adjacent_key(word: str, rng: random.Random, neighbours: dict[str, str]) -> str | None:
    """Replace one letter by a neighbouring key."""
    positions = [i for i in _typo_positions(word) if word[i].lower() in neighbours]
    if not positions:
        return None
    i = rng.choice(positions)
    new = rng.choice(neighbours[word[i].lower()])
    if word[i].isupper():
        new = new.upper()
    return word[:i] + new + word[i + 1 :]


def transposition(word: str, rng: random.Random) -> str | None:
    """Swap two adjacent, different letters."""
    ok = set(_typo_positions(word))
    positions = [i for i in ok if i + 1 in ok and word[i] != word[i + 1]]
    if not positions:
        return None
    i = rng.choice(sorted(positions))
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def deletion(word: str, rng: random.Random) -> str | None:
    """Drop one letter (never one that would leave a dangling apostrophe: can't -> can')."""
    positions = [i for i in _typo_positions(word) if (word[:i] + word[i + 1 :])[-1].isalpha()]
    if not positions:
        return None
    i = rng.choice(positions)
    return word[:i] + word[i + 1 :]


def doubled_letter(word: str, rng: random.Random) -> str | None:
    """Type one letter twice."""
    positions = _typo_positions(word)
    if not positions:
        return None
    i = rng.choice(positions)
    return word[: i + 1] + word[i] + word[i + 1 :]


# --------------------------------------------------------------------------
# English corruptions
# --------------------------------------------------------------------------

EN_CONFUSIONS = (
    ("their", "there", "they're"),
    ("your", "you're"),
    ("its", "it's"),
    ("then", "than"),
    ("from", "form"),
    ("lose", "loose"),
)


def en_confusion(word: str, rng: random.Random) -> str | None:
    """Swap a word for one it is commonly confused with (their/there...)."""
    return _swap_within_groups(word, rng, EN_CONFUSIONS)


# --------------------------------------------------------------------------
# French corruptions
# --------------------------------------------------------------------------

ACCENT_BASE = {
    "à": "a", "â": "a", "ä": "a",
    "é": "e", "è": "e", "ê": "e", "ë": "e",
    "î": "i", "ï": "i",
    "ô": "o", "ö": "o",
    "ù": "u", "û": "u", "ü": "u",
    "ç": "c",
}
E_ACCENTS = "éèê"

FR_HOMOPHONES = (
    ("a", "à"),
    ("et", "est"),
    ("ces", "ses", "c'est", "s'est"),
    ("son", "sont"),
    ("on", "ont"),
    ("ou", "où"),
    ("ce", "se"),
)
# Also English words: swapping them inside a code-switched sentence would
# corrupt the English half with a French word.
AMBIGUOUS_IN_MIXED = ("a", "on", "son")

VERB_ENDINGS = ("er", "ez", "é")
# Common words with a verb-like ending that are not verbs.
NOT_VERBS = frozenset(
    "chez assez nez hier hiver février janvier premier dernier cher mer fer "
    "été café marché côté clé étranger déjeuner after computer server manager "
    "never other together earlier".split()
)

# Common words ending in -s that are not plurals.
NOT_PLURALS = frozenset(
    "très plus dans sans vous nous mais jamais toujours alors puis depuis temps "
    "pays fois corps bras dessus dessous parfois longtemps tous sous vers hors "
    "près après moins ailleurs plusieurs français anglais mois trois bois fils "
    "sens cours avis repas gros gris dehors volontiers "
    "this always perhaps thus across unless".split()
)


def accent_strip(word: str, rng: random.Random) -> str | None:
    """Remove every accent from the word (élève -> eleve, à -> a)."""
    stripped = "".join(match_case(ch, ACCENT_BASE.get(ch.lower(), ch)) for ch in word)
    return stripped if stripped != word else None


def accent_swap(word: str, rng: random.Random) -> str | None:
    """Put the wrong accent on one e (é <-> è <-> ê)."""
    positions = [i for i, ch in enumerate(word) if ch.lower() in E_ACCENTS]
    if not positions:
        return None
    i = rng.choice(positions)
    new = rng.choice([a for a in E_ACCENTS if a != word[i].lower()])
    return word[:i] + match_case(word[i], new) + word[i + 1 :]


def fr_homophone(word: str, rng: random.Random, skip: tuple[str, ...] = ()) -> str | None:
    """Swap a grammatical homophone (a/à, et/est, ces/ses/c'est/s'est...)."""
    return _swap_within_groups(word, rng, FR_HOMOPHONES, skip)


def verb_ending(word: str, rng: random.Random) -> str | None:
    """Swap the homophone verb endings -er / -é / -ez."""
    low = word.lower()
    if len(word) < 4 or word.isupper() or low in NOT_VERBS:
        return None
    for ending in VERB_ENDINGS:
        if low.endswith(ending):
            new = rng.choice([e for e in VERB_ENDINGS if e != ending])
            return word[: -len(ending)] + new
    return None


def dropped_plural(word: str, rng: random.Random) -> str | None:
    """Drop a final plural -s."""
    low = word.lower()
    last_part = low.rsplit("-", 1)[-1]
    if len(last_part) < 4 or not low.endswith("s") or low.endswith("ss"):
        return None
    if last_part in NOT_PLURALS:
        return None
    return word[:-1]


# --------------------------------------------------------------------------
# Phonetic spellings: the word written the way it sounds, by someone who does
# not know how it is spelled. Unlike a typo, the result can be several letters
# away from the real word.
# --------------------------------------------------------------------------

MIN_PHONETIC_LETTERS = 4
_V = "aeiouyéèêàâîôû"  # French vowels
_C = "b-df-hj-np-tv-xz"  # consonants

# (pattern, replacement) pairs, applied to the lowercased word.
# Safe in both languages, so these are the only ones used in mixed sentences.
SHARED_PHONETIC = (
    (r"ph", "f"),  # pharmacy -> farmacy
    (r"([bdfmnprt])\1", r"\1"),  # arrivés -> arivés, offer -> ofer
    (r"(?<!s)tion", "sion"),  # attention -> attension
    (rf"(?<=[{_C}])y(?=[{_C}])", "i"),  # system -> sistem
)

EN_PHONETIC = (
    (r"tion\b", "shun"),  # station -> stashun
    (r"(?<!e)ight", "ite"),  # night -> nite
    (r"ck", "k"),  # back -> bak
    (r"\bwh(?=[aeiy])", "w"),  # which -> wich
    (r"\bwr", "r"),  # wrong -> rong
    (r"(?<=\w)ie(?=\w)", "ei"),  # friend -> freind
    (r"(?<=\w)ei(?=\w)", "ie"),  # their -> thier
    (rf"(?<=[{_C}])le\b", "el"),  # people -> peopel
    (r"ous\b", "us"),  # famous -> famus
    (r"(?<!s)c(?=[eiy])", "s"),  # decision -> desision
    (r"ould\b", "ood"),  # would -> wood
    (r"ee(?=\w)", "ea"),  # weekend -> weakend
    (r"ful\b", "full"),  # careful -> carefull
    (r"(?<=\w\w)ence\b", "ance"),  # experience -> experiance
    (r"(?<=\w\w)ance\b", "ence"),  # distance -> distence
    (r"(?<=\w{3})ent\b", "ant"),  # student -> studant
    (r"qu", "kw"),  # question -> kwestion
    (r"(?<=\w)x", "ks"),  # next -> nekst
    (r"([ls])\1", r"\1"),  # really -> realy, passport -> pasport
)

FR_PHONETIC = (
    (r"eau", "o"),  # bateau -> bato
    (r"(?<!e)au", "o"),  # aussi -> ossi
    (r"qu", "k"),  # quand -> kand
    (rf"(?<=[{_V}])ç", "ss"),  # reçu -> ressu
    (rf"(?<![{_V}])ç", "s"),  # garçon -> garson
    (rf"(?<=[{_V}])c(?=[eiyéèê])", "ss"),  # décidé -> déssidé
    (rf"(?<![{_V}s])c(?=[eiyéèê])", "s"),  # merci -> mersi
    (r"ai(?![lmn]|ent\b)", "è"),  # maison -> mèson
    (r"(?<=\w)(ais|ait|aient)\b", "é"),  # avait -> avé
    (r"ei(?![ln])", "è"),  # neige -> nège
    (r"(?<![iéy])en(?=[cdfstv]\w)", "an"),  # pendant -> pandant
    (r"(?<!i)an(?=[cdgst]\w)", "en"),  # vacances -> vacences
    (r"\bh", ""),  # hier -> ier
    (r"(?<=\w\w[aiouéèôûrn])(?<!en)[td]\b", ""),  # petit -> peti, grand -> gran
    (r"(?<=\w\wu)x\b", ""),  # mieux -> mieu
    (r"(?:(?<=[ae])|(?<=[eo]u))ill", "y"),  # travaille -> travaye
    (r"(?<!g)g(?=[eéèêi])", "j"),  # manger -> manjer
    (rf"(?<=[{_V}])s(?=[{_V}])", "z"),  # cuisine -> cuizine
    (rf"(ain|ein)(?![{_V}])", "in"),  # demain -> demin
    (r"th", "t"),  # bibliothèque -> bibliotèque
    (r"(?<!i)ll", "l"),  # nouvelle -> nouvele
)


def _compile(rules):
    return tuple((re.compile(pattern), repl) for pattern, repl in rules)


PHONETIC_RULES = {
    "en": _compile(SHARED_PHONETIC + EN_PHONETIC),
    "fr": _compile(SHARED_PHONETIC + FR_PHONETIC),
    "mixed": _compile(SHARED_PHONETIC),
}


def _phonetic_variants(word: str, rules) -> list[str]:
    """Every spelling one rule application away from `word`."""
    variants = {
        word[: m.start()] + m.expand(repl) + word[m.end() :]
        for pattern, repl in rules
        for m in pattern.finditer(word)
    }
    return sorted(variants - {word})


def phonetic(word: str, rng: random.Random, rules) -> str | None:
    """Respell one or two sounds of the word the way they are heard."""
    low = word.lower()
    if sum(ch.isalpha() for ch in word) < MIN_PHONETIC_LETTERS or word[1:] != low[1:]:
        return None
    new = low
    for _ in range(rng.choice((1, 2))):
        options = [v for v in _phonetic_variants(new, rules) if v != low]
        if not options:
            break
        new = rng.choice(options)
    return match_case(word, new) if new != low else None


# --------------------------------------------------------------------------
# Building examples
# --------------------------------------------------------------------------


def corruptions_for(lang: str, layout: str) -> dict:
    """Corruption functions that apply to `lang`, keyed by error type name."""
    table = {
        "adjacent_key": partial(adjacent_key, neighbours=key_neighbours(layout)),
        "transposition": transposition,
        "deletion": deletion,
        "doubled_letter": doubled_letter,
        "phonetic": partial(phonetic, rules=PHONETIC_RULES[lang]),
    }
    if lang in ("en", "mixed"):
        table["en_confusion"] = en_confusion
    if lang in ("fr", "mixed"):
        table["accent_strip"] = accent_strip
        table["accent_swap"] = accent_swap
        table["dropped_plural"] = dropped_plural
    if lang == "fr":
        table["fr_homophone"] = fr_homophone
        table["verb_ending"] = verb_ending  # skipped in mixed: would hit English -er words
    if lang == "mixed":
        table["fr_homophone"] = partial(fr_homophone, skip=AMBIGUOUS_IN_MIXED)
    return table


TYPO_TYPES = ("adjacent_key", "transposition", "deletion", "doubled_letter")


def _shuffled_types(corruptions: dict, rng: random.Random) -> list[str]:
    """Error types in the order to try them. Typos apply to almost any word, so
    half of the time the language-specific types get to go first; otherwise
    they would be rare in the eval set."""
    typos = [name for name in corruptions if name in TYPO_TYPES]
    specific = [name for name in corruptions if name not in TYPO_TYPES]
    rng.shuffle(typos)
    rng.shuffle(specific)
    return specific + typos if rng.random() < 0.5 else typos + specific


def inject_error(tokens: list[str], corruptions: dict, rng: random.Random, taken: set[int]):
    """Corrupt one not-yet-corrupted token in place. Returns the error record, or None."""
    names = _shuffled_types(corruptions, rng)
    indices = [i for i in range(len(tokens)) if i not in taken]
    for name in names:
        rng.shuffle(indices)
        for i in indices:
            prefix, core, suffix = split_token(tokens[i])
            if not is_word(core):
                continue
            new = corruptions[name](core, rng)
            if new and new != core:
                tokens[i] = prefix + new + suffix
                return {"word_index": i, "type": name, "original": core, "corrupted": new}
    return None


def corrupt_sentence(sentence: str, corruptions: dict, rng: random.Random, n_errors: int):
    """Return (corrupted sentence, error records) with up to `n_errors` errors."""
    tokens = sentence.split()
    errors = []
    for _ in range(n_errors):
        error = inject_error(tokens, corruptions, rng, {e["word_index"] for e in errors})
        if error is None:
            break
        errors.append(error)
    errors.sort(key=lambda e: e["word_index"])
    return " ".join(tokens), errors


def build_examples(
    sentences: dict[str, list[str]],
    seed: int = 42,
    clean_ratio: float = 0.3,
    layout: str = "qwertz-ch",
    variants: int = 1,
) -> list[dict]:
    """Turn clean sentences into eval examples, in a seeded shuffled order."""
    rng = random.Random(seed)
    examples = []
    for lang in LANGS:
        corruptions = corruptions_for(lang, layout)
        items = [s for s in sentences.get(lang, []) for _ in range(variants)]
        keep_clean = set(rng.sample(range(len(items)), round(len(items) * clean_ratio)))
        for n, sentence in enumerate(items):
            target = " ".join(sentence.split())
            corrupted, errors = target, []
            if n not in keep_clean:
                n_errors = rng.choices((1, 2, 3), weights=(6, 3, 1))[0]
                corrupted, errors = corrupt_sentence(target, corruptions, rng, n_errors)
            examples.append(
                {
                    "id": f"{lang}-{n:04d}",
                    "lang": lang,
                    "input": corrupted,
                    "target": target,
                    "clean": not errors,
                    "errors": errors,
                }
            )
    rng.shuffle(examples)  # so that `baseline.py --limit N` sees every language
    return examples


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

PLACEHOLDERS = {
    "en": [
        "I left my keys on the kitchen table before going to work this morning.",
        "Their house is much bigger than the one we visited last week.",
        "If you lose your ticket, you will have to buy another one at the station.",
        "The dog wags its tail whenever it's time for a walk.",
        "Please fill in the form and send it back before Friday.",
        "We drove from Geneva to Lyon in less than two hours.",
        "There are three reasons why the meeting was postponed.",
        "She finished the report first, then she called the client.",
        "You're going to need your passport to cross the border.",
        "They're still waiting for the results of the experiment.",
        "The screws on this shelf are loose, so be careful.",
        "Could you tell me where the nearest pharmacy is?",
        "The weather forecast says it will rain all weekend.",
        "He didn't know whether to laugh or to apologise.",
        "My brother's new apartment has a wonderful view of the lake.",
        "I would rather walk than take the bus in this traffic.",
        "The committee has not yet made its decision about the budget.",
        "Don't forget to back up your files before you update the system.",
        "Most of the students handed in their essays on time.",
        "It's been a long day, and I can't wait to get home.",
    ],
    "fr": [
        "Il a oublié ses clés sur la table de la cuisine ce matin.",
        "Nous sommes allés à la bibliothèque pour rendre les livres empruntés.",
        "Elle est très contente parce que ses parents sont arrivés hier soir.",
        "Tu peux choisir le thé ou le café, mais je ne sais pas où sont les tasses.",
        "Mon grand-père a fêté ses nonante ans au mois de février.",
        "Le billet coûte septante francs et il faut le payer à l'avance.",
        "C'est la première fois que ces élèves visitent un musée.",
        "Il s'est levé tôt pour préparer le petit déjeuner des enfants.",
        "On a décidé de rester à la maison car ils ont annoncé de la neige.",
        "Vous devez envoyer ce formulaire avant la fin de la semaine.",
        "Les enfants ont mangé toutes les pommes que j'avais achetées au marché.",
        "Son frère et sa sœur sont partis étudier à l'étranger.",
        "Je me demande si ce restaurant est encore ouvert après minuit.",
        "Elle se promène souvent le long du lac quand il fait beau.",
        "Pouvez-vous m'expliquer comment arriver à la gare, s'il vous plaît ?",
        "Les voisins ont réparé la fenêtre cassée pendant les vacances.",
        "J'aimerais réserver une chambre pour trois nuits à Genève.",
        "Ce garçon a déjà terminé ses devoirs, alors il peut aller jouer dehors.",
        "Nous avons reçu votre lettre et nous allons vous répondre bientôt.",
        "Il faut arroser les plantes tous les deux jours pendant l'été.",
    ],
    "mixed": [
        "J'ai un meeting à quatorze heures, so I can't join you for lunch.",
        "Can you send me the slides avant la réunion de demain matin ?",
        "Le deadline est vendredi, but I think we can finish earlier.",
        "I talked to the manager et elle est d'accord avec notre proposition.",
        "On se retrouve au café, then we can walk to the office together.",
        "The server is down again, il faut redémarrer la machine tout de suite.",
        "Merci pour ton message, I will call you back après le déjeuner.",
        "She said the report is ready, mais il manque encore les chiffres de février.",
        "Je dois push my changes before the review, sinon ça va bloquer les autres.",
        "We need septante chairs for the event, et aussi quelques tables.",
        "Tu peux regarder tes mails, I sent you the file this morning.",
        "The weather is lovely today, on pourrait manger dehors à midi.",
        "Il m'a dit que their flight was delayed à cause de la neige.",
        "I forgot my badge à la maison, so I had to wait at the reception.",
        "Ce bug est vraiment étrange, it only happens when the cache is empty.",
        "Let me know si tu as besoin d'aide pour préparer la présentation.",
        "Les tests passent en local, but the pipeline still fails on the server.",
        "My sister lives in Lausanne et elle travaille à l'hôpital depuis trois ans.",
        "Bonne nouvelle, the client accepted our offer sans demander de réduction.",
        "I'll be there in ten minutes, garde-moi une place près de la fenêtre.",
    ],
}


def sentence_path(data_dir: Path, lang: str) -> Path:
    return data_dir / f"sentences_{lang}.txt"


def ensure_sentence_files(data_dir: Path) -> list[Path]:
    """Create placeholder sentence files for any that are missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for lang in LANGS:
        path = sentence_path(data_dir, lang)
        if path.exists():
            continue
        header = f"{PLACEHOLDER_MARK} sentences ({lang}) - replace this whole file with your own writing, one sentence per line."
        path.write_text("\n".join([header, *PLACEHOLDERS[lang]]) + "\n", encoding="utf-8")
        created.append(path)
    return created


def load_sentences(path: Path) -> list[str]:
    """Read one sentence per line, skipping blank lines and # comments."""
    lines = unicodedata.normalize("NFC", path.read_text(encoding="utf-8")).splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def is_placeholder(path: Path) -> bool:
    return path.read_text(encoding="utf-8").startswith(PLACEHOLDER_MARK)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(examples: list[dict], out: Path) -> None:
    print(f"Wrote {len(examples)} examples to {out}")
    for lang in LANGS:
        rows = [e for e in examples if e["lang"] == lang]
        clean = sum(e["clean"] for e in rows)
        print(f"  {lang:<6} {len(rows):>4} examples, {clean} clean")
    types = Counter(err["type"] for e in examples for err in e["errors"])
    print("  errors: " + ", ".join(f"{name}={n}" for name, n in sorted(types.items())))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out", type=Path, default=None, help="default: <data-dir>/eval.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean-ratio", type=float, default=0.3, help="share of examples left clean")
    p.add_argument("--layout", choices=sorted(KEYBOARD_ROWS), default="qwertz-ch")
    p.add_argument("--variants", type=int, default=1, help="examples generated per sentence")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out = args.out or args.data_dir / "eval.jsonl"
    for path in ensure_sentence_files(args.data_dir):
        print(f"Created placeholder file {path}")
    sentences = {}
    for lang in LANGS:
        path = sentence_path(args.data_dir, lang)
        sentences[lang] = load_sentences(path)
        if is_placeholder(path):
            print(f"WARNING: {path} still contains placeholder sentences.")
    examples = build_examples(sentences, args.seed, args.clean_ratio, args.layout, args.variants)
    write_jsonl(out, examples)
    print_summary(examples, out)


if __name__ == "__main__":
    main()
