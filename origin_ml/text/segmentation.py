"""Deterministic rule-based sentence segmentation with exact character offsets.

Design constraints (SPEC AD-6):

- No model dependency, fully deterministic.
- Every sentence carries ``(start, end)`` character offsets into the original
  document such that ``text[start:end] == sentence.text`` (round-trip property,
  SPEC L-1).

Behavior:

- Paragraph breaks (a newline followed by an optionally-indented blank line) are
  always sentence boundaries.
- Within a paragraph, sentences end at runs of ``.``, ``!``, ``?`` (optionally
  followed by closing quotes/brackets) that are followed by whitespace or the
  paragraph end.
- A single ``.`` does not end a sentence when the preceding word is a known
  abbreviation (``Dr.``, ``etc.``) or a single letter (initials such as ``J.``,
  and thereby dotted initialisms such as ``U.S.`` or ``e.g.``).
- Decimal numbers (``3.14``) never split because the ``.`` is not followed by
  whitespace.
- Single newlines are treated as ordinary whitespace: a sentence may wrap
  across lines.

Known, accepted ambiguity: a sentence that genuinely ends with an abbreviation
("... in the U.S. Next year ...") merges with its successor. This is the
standard trade-off of deterministic segmenters and is documented rather than
special-cased.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["SentenceSpan", "segment_sentences"]


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """A sentence with exact character offsets into the source document."""

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span ({self.start}, {self.end})")


_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_TERMINATOR_RUN = re.compile(r"[.!?]+[\"'）)\]”’»]*")
# The last alphabetic word (possibly containing internal apostrophes) directly
# before a terminating dot, used for abbreviation checks.
_WORD_BEFORE_DOT = re.compile(r"([A-Za-z][A-Za-z']*)\.\Z")

# Lowercase abbreviations that do not end a sentence when followed by ".".
# Single letters (initials / dotted initialisms) are handled separately.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "rev",
        "hon",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "fig",
        "figs",
        "eq",
        "sec",
        "chap",
        "al",
        "cf",
        "resp",
        "approx",
        "dept",
        "est",
        "vol",
        "pp",
        "ed",
        "eds",
        "inc",
        "ltd",
        "co",
        "corp",
        "univ",
        "assn",
        "bros",
        "ave",
        "blvd",
        "rd",
        "mt",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "sat",
        "sun",
    }
)


def _is_abbreviation_before(text: str, dot_end: int) -> bool:
    """Return True if the ``.`` ending at ``dot_end`` follows an abbreviation.

    ``dot_end`` is the index one past the dot. Single-letter words count as
    initials, which also covers dotted initialisms ("U.S.", "e.g.", "Ph.D.")
    because their final component is a single letter.
    """
    match = _WORD_BEFORE_DOT.search(text, 0, dot_end)
    if match is None:
        return False
    word = match.group(1)
    return len(word) == 1 or word.lower() in _ABBREVIATIONS


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        spans.append((pos, match.start()))
        pos = match.end()
    spans.append((pos, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _boundaries_in_paragraph(text: str, start: int, end: int) -> list[int]:
    """Sentence end positions (exclusive, pre-trim) within ``text[start:end]``."""
    boundaries: list[int] = []
    for match in _TERMINATOR_RUN.finditer(text, start, end):
        run_end = match.end()
        if run_end < end and not text[run_end].isspace():
            continue  # e.g. mid-token punctuation such as "3.14" or "example.com"
        terminators = match.group().rstrip("\"'）)]”’»")
        if terminators == "." and _is_abbreviation_before(text, match.start() + 1):
            continue
        boundaries.append(run_end)
    if not boundaries or boundaries[-1] < end:
        boundaries.append(end)
    return boundaries


def segment_sentences(text: str) -> list[SentenceSpan]:
    """Split ``text`` into sentences with offsets that round-trip exactly."""
    sentences: list[SentenceSpan] = []
    for para_start, para_end in _paragraph_spans(text):
        cursor = para_start
        for boundary in _boundaries_in_paragraph(text, para_start, para_end):
            segment = text[cursor:boundary]
            if segment.strip():
                lead = len(segment) - len(segment.lstrip())
                trail = len(segment) - len(segment.rstrip())
                sent_start = cursor + lead
                sent_end = boundary - trail
                sentences.append(
                    SentenceSpan(text=text[sent_start:sent_end], start=sent_start, end=sent_end)
                )
            cursor = boundary
    return sentences
