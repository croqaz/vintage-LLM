"""Conservative OCR cleanup for seed text before it is sent to the model.

Design principle: only touch what is *unambiguously* noise. We fix structural
artefacts (hyphen line-breaks, stray newlines, spaced punctuation) and strip a
small, hand-picked set of junk glyphs — but we deliberately PRESERVE legitimate
typography that also shows up as "non-ascii": em dashes, accented letters,
ligatures, currency and section signs (— é £ æ œ § « » etc.). Garbled *words*
(e.g. "ConfutcUion", "s°nsual") are left for the model's prompt to mend,
since we cannot repair them without risking real text.

Use ``clean(text)`` for the default profile, or ``clean(text, aggressive=True)``
to also drop stray Greek/box glyphs that are usually mis-scans.
"""

from __future__ import annotations

import re
import unicodedata

# Glyphs that are essentially never legitimate in a clean English/period text
# and are almost always OCR speckle. NOTE: em dash, accents, ligatures, £, §,
# guillemets are intentionally NOT here.
_JUNK_CHARS = '^|•~¬■□▪●◦♦†‡¤¢∎_'

# In aggressive mode, also strip isolated Greek/Cyrillic letters and a few more
# marks that are usually mis-recognised Latin. Kept separate so it is opt-in.
_AGGRESSIVE_EXTRA = '{}\\<>'

_JUNK_RE = re.compile('[' + re.escape(_JUNK_CHARS) + ']')
_AGGRESSIVE_RE = re.compile('[' + re.escape(_JUNK_CHARS + _AGGRESSIVE_EXTRA) + ']')

# " word- \n next " -> " word next "  (join words split across a line break)
_HYPHEN_LINEBREAK_RE = re.compile(r'(\w)[-¬]\s*\n\s*(\w)')
# Space(s) before common punctuation:  " word ;"  ->  "word;"
_SPACE_BEFORE_PUNCT_RE = re.compile(r'\s+([;:,.!?])')
# Collapse any run of whitespace (incl. the newlines we don't otherwise need)
_WS_RE = re.compile(r'\s+')


def clean(text: str, *, aggressive: bool = False, keep_paragraphs: bool = False) -> str:
    """Return a conservatively-cleaned copy of ``text``.

    Parameters
    ----------
    aggressive:
        Also strip brace/angle glyphs commonly produced by bad scans.
    keep_paragraphs:
        Preserve blank-line paragraph breaks (collapse only intra-paragraph
        whitespace). When False (default) the whole chunk becomes one flowed
        block, which is usually what a "continue"/"rewrite" prompt wants.
    """
    if not text:
        return ''

    # 1. Normalise unicode (folds compatibility forms, e.g. ﬁ -> fi).
    text = unicodedata.normalize('NFKC', text)

    # 2. Join words broken by a hyphen at a line end (do this before we touch
    #    newlines, and loop because chains like "a-\nb-\nc" need two passes).
    prev = None
    while prev != text:
        prev = text
        text = _HYPHEN_LINEBREAK_RE.sub(r'\1\2', text)

    # 3. Drop junk glyphs.
    text = (_AGGRESSIVE_RE if aggressive else _JUNK_RE).sub('', text)

    # 4. Fix spaced-out punctuation (" ;" -> ";").
    text = _SPACE_BEFORE_PUNCT_RE.sub(r'\1', text)

    # 5. Whitespace handling.
    if keep_paragraphs:
        paras = re.split(r'\n\s*\n', text)
        text = '\n\n'.join(_WS_RE.sub(' ', p).strip() for p in paras if p.strip())
    else:
        text = _WS_RE.sub(' ', text).strip()

    return text


if __name__ == '__main__':  # tiny manual demo
    import sys

    sample = (
        sys.stdin.read()
        if not sys.stdin.isatty()
        else (
            'the same general char-\nacter as the Pro^mium itself ; and along\n'
            "M'ith these he restates the con-\nclusion — a s°nsual affair."
        )
    )
    print(clean(sample))
