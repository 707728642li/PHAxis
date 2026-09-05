"""Journal-facing text contracts shared by PHAxis manuscript builders.

The functions in this module are deliberately small and dependency-free so
that the Markdown compiler and the DOCX assembler can independently enforce
the same submission rule.  They operate on the final, token-resolved text;
machine placeholders therefore cannot hide an over-length abstract.
"""

from __future__ import annotations

import re


# Plant Phenomics states a 250-word ceiling.  PHAxis deliberately enforces the
# stricter, unambiguous submission contract ``word_count < 250`` so no journal
# counter or last-minute token resolution can leave the manuscript on the
# boundary.
ABSTRACT_WORD_LIMIT = 249

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_KEYWORDS_LINE = re.compile(
    r"^\s*(?:\*\*|__)?keywords(?:\*\*|__)?\s*:\s*",
    flags=re.IGNORECASE,
)
_WORD = re.compile(
    r"[^\W_]+(?:[\u0027\u2019\u2010-\u2015\u207b-][^\W_]+)*",
    flags=re.UNICODE,
)


class ManuscriptTextContractError(ValueError):
    """The manuscript text cannot satisfy the journal-facing contract."""


def extract_abstract(markdown: str) -> str:
    """Return the single non-empty level-two ``Abstract`` section.

    The section ends at the next level-one or level-two heading or at the
    conventional bold ``Keywords:`` line.  Lower-level headings are retained
    as text so an accidental subheading cannot make part of an abstract
    disappear from the count; journal keywords never consume the Abstract
    word budget.
    """

    if not isinstance(markdown, str):
        raise ManuscriptTextContractError("manuscript Markdown must be text")
    lines = markdown.splitlines()
    starts: list[int] = []
    for index, line in enumerate(lines):
        match = _HEADING.fullmatch(line.strip())
        if (
            match is not None
            and len(match.group(1)) == 2
            and match.group(2).strip().casefold() == "abstract"
        ):
            starts.append(index)
    if len(starts) != 1:
        raise ManuscriptTextContractError(
            f"manuscript must contain exactly one level-two Abstract heading; found {len(starts)}"
        )
    start = starts[0] + 1
    end = len(lines)
    for index in range(start, len(lines)):
        match = _HEADING.fullmatch(lines[index].strip())
        if match is not None and len(match.group(1)) <= 2:
            end = index
            break
        if _KEYWORDS_LINE.match(lines[index]):
            end = index
            break
    abstract = "\n".join(lines[start:end]).strip()
    if not abstract:
        raise ManuscriptTextContractError("Abstract section is empty")
    return abstract


def abstract_word_count(markdown: str) -> int:
    """Count words in the final abstract with deterministic Unicode rules."""

    abstract = extract_abstract(markdown)
    return text_word_count(abstract)


def text_word_count(text: str) -> int:
    """Count words in arbitrary text with the manuscript's exact rules."""

    if not isinstance(text, str):
        raise ManuscriptTextContractError("word-count input must be text")
    # Count linked labels, not destination URLs.  The remaining substitutions
    # remove Markdown presentation marks without joining adjacent words.
    plain = _MARKDOWN_LINK.sub(r"\1", text)
    plain = re.sub(r"[`*_~]", "", plain)
    return len(_WORD.findall(plain))


def require_abstract_within_limit(
    markdown: str,
    *,
    limit: int = ABSTRACT_WORD_LIMIT,
) -> int:
    """Return the count or raise when the final abstract exceeds ``limit``."""

    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ManuscriptTextContractError("abstract word limit must be a positive integer")
    count = abstract_word_count(markdown)
    if count > limit:
        raise ManuscriptTextContractError(
            f"Plant Phenomics abstract word limit exceeded: {count} > {limit}"
        )
    return count


__all__ = [
    "ABSTRACT_WORD_LIMIT",
    "ManuscriptTextContractError",
    "abstract_word_count",
    "extract_abstract",
    "require_abstract_within_limit",
    "text_word_count",
]
