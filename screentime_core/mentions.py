"""Brand mentions and spoken sponsorship disclosures in a transcript: a port of Screen-Time's src/lib/metrics/mentions.ts.

Turns are objects or dicts with ``start_sec``, ``end_sec`` and ``text``. Turns carry no word timings, so a match's time
is estimated by its word position inside its turn, and every Mention returned has approximate=True.

Matching works on tokens: runs of letters, digits and combining marks (an apostrophe followed by letters stays inside
the word), normalized with NFKD, marks removed, lowercased, a possessive "'s" dropped and other apostrophes removed.
Python's ``re`` has no \\p{L}, so the TypeScript's regular expression is a scanner over unicodedata categories here; token
offsets are code-point offsets (JavaScript's are UTF-16), which slice to the same text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple, Optional

from .errors import ScreentimeError
from .models import Mention

_APOSTROPHES = "'\u2019"
# What JavaScript's String.prototype.trim removes: its WhiteSpace and LineTerminator characters. Python's str.strip()
# also removes \x1c-\x1f and \x85 and keeps \ufeff, so the set is spelled out.
_JS_SPACE = "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_SENTENCE_END = re.compile(r"[.!?]+")


class Token(NamedTuple):
    norm: str
    start: int  # code-point offsets into the original text
    end: int


def tokenize(text: str) -> list[Token]:
    """Lowercased, accent-free word tokens; hyphens and other punctuation split words, possessives are dropped.

    The scanner reads what the TypeScript's /[\\p{L}\\p{N}\\p{M}]+(?:['\u2019][\\p{L}\\p{M}]+)*/gu matches: combining
    marks stay inside a word, so a decomposed accent does not split it.
    """
    tokens: list[Token] = []
    n = len(text)
    i = 0
    while i < n:
        if not _is_word_char(text[i]):
            i += 1
            continue
        start = i
        while i < n and _is_word_char(text[i]):
            i += 1
        while i + 1 < n and text[i] in _APOSTROPHES and _is_letter_or_mark(text[i + 1]):
            i += 2
            while i < n and _is_letter_or_mark(text[i]):
                i += 1
        norm = _normalize_word(text[start:i])
        if norm:
            tokens.append(Token(norm, start, i))
    return tokens


def distinct_names(names: Iterable[str]) -> list[str]:
    """One spelling per name as find_mentions matches them, keeping the first: "LaCroix" and "La Croix" are one name.

    Use it for transcription keywords: given both spellings, Voice Transcribe wrote one spoken "LaCroix" as
    "La Croix LaCroix".
    """
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if not isinstance(name, str):
            continue
        key = _joined(name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def find_mentions(turns: Iterable[Any], aliases: Iterable[str]) -> list[Mention]:
    """Brand mentions in timed transcript turns, sorted by time.

    Case-insensitive and whole-word, ignoring punctuation, accents and possessives, with spaced, hyphenated and joined
    spellings the same ("Coca-Cola", "coca cola", "CocaCola"); "&" also matches a spoken "and". The longest alias at a
    position wins. A name that follows the same name with only a space between is one mention ("La Croix LaCroix": one
    spoken name the transcript wrote in both spellings); "Nike, Nike" and two different names side by side are two.
    """
    spellings: list[str] = []
    for alias in aliases:
        if not isinstance(alias, str):
            continue
        spellings.append(alias)
        if "&" in alias:  # "H&M" is heard as "H and M"
            spellings.append(alias.replace("&", " and "))
    terms = list(dict.fromkeys(j for j in (_joined(s) for s in spellings) if j))
    if not terms:
        return []
    mentions: list[Mention] = []
    for turn in turns:
        start_sec, end_sec, text = _turn(turn)
        tokens = tokenize(text)
        # The previous match in this turn: its first token, the token after its last, and the name it matched.
        prev: Optional[tuple[int, int, str]] = None
        i = 0
        while i < len(tokens):
            length = _match_at(tokens, i, terms)
            if length == 0:
                i += 1
                continue
            name = "".join(t.norm for t in tokens[i : i + length])
            # The same name right after the previous one with only a space between: written twice, so that mention
            # grows. A different name next to it ("LaCroix Pamplemousse") is its own mention.
            again = (
                prev is not None
                and prev[1] == i
                and prev[2] == name
                and not text[tokens[i - 1].end : tokens[i].start].strip(_JS_SPACE)
            )
            first = prev[0] if again and prev is not None else i
            if again:
                mentions.pop()
            mentions.append(_to_mention(start_sec, end_sec, text, tokens, first, i + length - first))
            prev = (first, i + length, name)
            i += length
    return sorted(mentions, key=lambda m: m.start_sec)


# ----------------------------------------------------------------------------------------------------------------------
# Spoken disclosures


def _patterns(specs: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Word sequences after tokenize: "a|b" are alternatives, a trailing "?" makes a word optional, and IS is "is" or
    nothing right after a word written with "'s" (tokenize reads "it's" as the token "it")."""
    return tuple(tuple(spec.split(" ")) for spec in specs)


# Patterns that open a clause of their own. A negation inside one stops it matching ("this is not an ad", "this isn't an
# ad"), so a negation before one belongs to something else ("that's actually not bad this is an ad"), unless a verb that
# takes the clause comes between ("I don't think this is an ad").
_CLAUSE_PATTERNS = _patterns(
    [
        "this|it|that video|post|reel|clip|one? IS an|a paid? ad",
        "this|it|that video|post|reel|clip|one? IS a? paid? sponsored",
    ]
)
# Patterns without a subject, which a negation up to three words before denies ("not a paid partnership").
_PHRASE_PATTERNS = _patterns(["hashtag ad|sponsored", "sponsored by", "paid partnership", "partnered with", "in partnership with"])
# Words that turn a following disclosure into a denial ("this is not sponsored", "no ads").
_NEGATIONS = frozenset({"not", "no", "isnt", "wasnt", "arent", "werent", "never", "without", "dont", "doesnt", "nt"})
_NEGATION_WINDOW = 3
# Verbs that carry a negation before them onto the clause they take: "I don't think this is an ad".
_CLAUSE_VERBS = frozenset({"think", "believe", "say", "saying", "said", "mean", "means", "know", "sure"})
# "thanks to Alani for sponsoring": how far back from "for sponsoring" the "thanks" may be.
_THANKS_WINDOW = 8


def find_spoken_disclosures(turns: Iterable[Any]) -> list[Mention]:
    """Where a transcript discloses the sponsorship, sorted by time.

    Finds "#ad", "hashtag ad", "this is an ad", "this video is an ad", "Ad." on its own, "sponsored by", "this is
    sponsored", "paid partnership", "partnered with", "in partnership with", "thanks to X for sponsoring". Never a word
    inside another ("add", "had", "advert"), never a passing "I saw an ad" or "no ads", and never a denial: a negation
    inside the statement ("this isn't an ad") or, for the phrases without a subject, up to three words earlier in the
    same sentence ("not a paid partnership") cancels it.
    """
    found: list[Mention] = []
    for turn in turns:
        start_sec, end_sec, text = _turn(turn)
        tokens = tokenize(text)
        # Sentence number of each token, so a negation in the previous sentence does not count.
        sentence = [len(_SENTENCE_END.findall(text, 0, t.start)) for t in tokens]
        i = 0
        while i < len(tokens):
            length = _disclosure_at(tokens, i, text, sentence)
            if length and not _negated(tokens, i, text, sentence):
                found.append(_to_mention(start_sec, end_sec, text, tokens, i, length))
                i += length
            else:
                i += 1
    return sorted(found, key=lambda m: m.start_sec)


def _negated(tokens: Sequence[Token], i: int, text: str, sentence: Sequence[int]) -> bool:
    clause = any(_match_pattern(tokens, text, i, p) > 0 for p in _CLAUSE_PATTERNS)
    for j in range(max(0, i - _NEGATION_WINDOW), i):
        if sentence[j] != sentence[i] or tokens[j].norm not in _NEGATIONS:
            continue
        if not clause or any(t.norm in _CLAUSE_VERBS for t in tokens[j + 1 : i]):
            return True
    return False


def _disclosure_at(tokens: Sequence[Token], i: int, text: str, sentence: Sequence[int]) -> int:
    """Tokens a disclosure covers from tokens[i], or 0."""
    t = tokens[i]
    char_before = text[t.start - 1] if t.start > 0 else ""
    char_after = text[t.end] if t.end < len(text) else ""
    # A hyphen joins the word into another one ("ad-free", "self-sponsored").
    if char_before == "-" or char_after == "-":
        return 0
    # "#ad", "#sponsored" written into a caption-style transcript.
    if t.norm in ("ad", "sponsored") and char_before == "#":
        return 1
    # "Ad." as a sentence of its own.
    if t.norm == "ad":
        before = text[: t.start].rstrip(_JS_SPACE)
        after = text[t.end :].lstrip(_JS_SPACE)
        if (not before or before[-1] in ".!?:") and (not after or after[0] in ".!?"):
            return 1
    # "thanks to Alani for sponsoring": from "thanks"/"thank" to "sponsoring", within one sentence.
    if t.norm in ("thanks", "thank"):
        for j in range(i + 1, min(len(tokens), i + _THANKS_WINDOW + 2)):
            if sentence[j] != sentence[i]:
                break
            if tokens[j].norm == "sponsoring" and tokens[j - 1].norm == "for":
                return j - i + 1
    best = 0
    for pattern in _PHRASE_PATTERNS + _CLAUSE_PATTERNS:
        best = max(best, _match_pattern(tokens, text, i, pattern))
    # The match may end inside a hyphenated word ("this is an ad-free video").
    if best:
        end = tokens[i + best - 1].end
        if end < len(text) and text[end] == "-":
            return 0
    return best


def _match_pattern(tokens: Sequence[Token], text: str, i: int, pattern: Sequence[str], k: int = 0) -> int:
    """Tokens ``pattern`` covers from tokens[i], or -1 when it does not match there. Optional words are tried both ways."""
    if k == len(pattern):
        return 0
    slot = pattern[k]
    if slot == "IS":
        if i < len(tokens) and tokens[i].norm == "is":
            n = _match_pattern(tokens, text, i + 1, pattern, k + 1)
            if n >= 0:
                return n + 1
        # "it's", "this's": the previous word carried the verb.
        if i > 0 and _ends_with_apostrophe_s(text[tokens[i - 1].start : tokens[i - 1].end]):
            return _match_pattern(tokens, text, i, pattern, k + 1)
        return -1
    optional = slot.endswith("?")
    words = (slot[:-1] if optional else slot).split("|")
    if i < len(tokens) and tokens[i].norm in words:
        n = _match_pattern(tokens, text, i + 1, pattern, k + 1)
        if n >= 0:
            return n + 1
    return _match_pattern(tokens, text, i, pattern, k + 1) if optional else -1


# ----------------------------------------------------------------------------------------------------------------------
# Helpers


def _turn(turn: Any) -> tuple[float, float, str]:
    """(start_sec, end_sec, text) from a dict or an object with those attributes."""
    if isinstance(turn, Mapping):
        start, end, text = turn.get("start_sec"), turn.get("end_sec"), turn.get("text")
    else:
        start, end, text = getattr(turn, "start_sec", None), getattr(turn, "end_sec", None), getattr(turn, "text", None)
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (start, end)):
        raise ScreentimeError("bad_input", "A transcript turn needs start_sec and end_sec in seconds.")
    return float(start), float(end), "" if text is None else str(text)


def _to_mention(start_sec: float, end_sec: float, text: str, tokens: Sequence[Token], i: int, length: int) -> Mention:
    duration = max(0.0, end_sec - start_sec)

    def at(k: int) -> float:
        return start_sec + (duration * k) / len(tokens)

    return Mention(
        start_sec=at(i),
        end_sec=at(i + length),
        text=text[tokens[i].start : tokens[i + length - 1].end],
        approximate=True,
    )


def _match_at(tokens: Sequence[Token], i: int, terms: Sequence[str]) -> int:
    """Length in tokens of the longest term that starts at tokens[i], or 0. Joining whole tokens keeps it whole-word."""
    best = 0
    for term in terms:
        joined = ""
        j = i
        while j < len(tokens) and len(joined) < len(term):
            joined += tokens[j].norm
            if joined == term:
                best = max(best, j - i + 1)
            if not term.startswith(joined):
                break
            j += 1
    return best


def _joined(name: str) -> str:
    """A name as one run of normalized letters: "La Croix", "LaCroix" and "la-croix" are all "lacroix"."""
    return "".join(t.norm for t in tokenize(name))


def _normalize_word(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word)
    s = "".join(ch for ch in decomposed if not unicodedata.category(ch).startswith("M")).lower()
    if len(s) >= 2 and s[-1] == "s" and s[-2] in _APOSTROPHES:
        s = s[:-2]
    return s.replace("'", "").replace("\u2019", "")


def _ends_with_apostrophe_s(word: str) -> bool:
    return len(word) >= 2 and word[-2] in _APOSTROPHES and word[-1] in "sS"


def _is_word_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LNM"


def _is_letter_or_mark(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LM"
