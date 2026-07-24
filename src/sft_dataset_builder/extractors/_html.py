"""Shared HTML -> text helper used by the HTML and EPUB extractors.

Uses BeautifulSoup when available for higher fidelity, and falls back to a
stdlib ``html.parser`` implementation so HTML/EPUB still work without the extra.
"""

from __future__ import annotations

from html.parser import HTMLParser

# Block-level tags whose text becomes a separate unit/paragraph.
_BLOCK_TAGS = {
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "blockquote",
    "pre",
    "tr",
    "caption",
    "figcaption",
    "section",
    "article",
    "dd",
    "dt",
}
# Tags whose contents should be dropped entirely.
_SKIP_TAGS = {"script", "style", "noscript", "head", "template"}


def html_to_blocks(html: str) -> tuple[list[str], bool]:
    """Convert an HTML string to a list of text blocks.

    Returns ``(blocks, used_beautifulsoup)`` so callers can warn when the
    lower-fidelity stdlib fallback was used.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _stdlib_blocks(html), False

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(list(_SKIP_TAGS)):
        tag.decompose()

    blocks: list[str] = []
    for element in soup.find_all(sorted(_BLOCK_TAGS)):
        text = element.get_text(" ", strip=True)
        if text:
            blocks.append(text)
    if not blocks:  # no recognized block tags — fall back to whole-document text
        whole = soup.get_text("\n", strip=True)
        blocks = [line.strip() for line in whole.splitlines() if line.strip()]
    return blocks, True


class _BlockTextParser(HTMLParser):
    """Minimal stdlib HTML-to-text parser that emits block-level text chunks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._buf: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS or tag in ("br", "div"):
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self._buf.append(data.strip())

    def _flush(self) -> None:
        text = " ".join(self._buf).strip()
        if text:
            self.blocks.append(text)
        self._buf = []


def _stdlib_blocks(html: str) -> list[str]:
    parser = _BlockTextParser()
    parser.feed(html)
    parser.close()
    parser._flush()
    return parser.blocks
