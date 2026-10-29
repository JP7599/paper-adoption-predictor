"""Clean the abstracts we get from the APIs.

About 3.6 % of bioRxiv abstracts carry the markup of a graphical abstract, for example
"Graphic Abstract O_FIG O_LINKSMALLFIG WIDTH=200 HEIGHT=95 SRC="FIGDIR/small/630886v1_ufig1.gif" ALT="Figure 1"> View larger
version (30K): org.highwire.dtl.DTLVardef@1d88af9 ... M_FIG C_FIG". Those tokens are not words of the paper, so they are removed
before counting words, looking for code links, embedding or modelling. Mathematical "<" and ">" signs are left alone.
"""
from __future__ import annotations

import re

import pandas as pd

_FIGURE_BLOCK = re.compile(r"O_FIG.*?C_FIG", re.DOTALL)                    # a whole graphical-abstract block
_MARKUP_TOKENS = re.compile(r"\b[OCM]_[A-Z]+?(?=[A-Z][a-z]|[^A-Z]|$)|\S*highwire\.dtl\.DTLVardef@\w+|_HPS_FORMAT_FIGEXP|View (larger version \(\d+K\)|this table):?|"
                            r"\b(WIDTH|HEIGHT)=\d+\b|SRC=\"[^\"]*\"|ALT=\"[^\"]*\">?|\bFIGDIR\b")
_LABELS = re.compile(r"\b(Graphic(al)? Abstract|Highlights?|Abstract Figure)\b\s*:?", re.IGNORECASE)
_TAGS = re.compile(r"</?(p|br|i|b|sup|sub|em|strong|div|span|img)\b[^>]*>", re.IGNORECASE)
_SPACES = re.compile(r"\s+")


def clean_abstract(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = _FIGURE_BLOCK.sub(" ", text)
    text = _MARKUP_TOKENS.sub(" ", text)
    text = _TAGS.sub(" ", text)
    text = _LABELS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def clean_series(s: pd.Series) -> pd.Series:
    return s.map(clean_abstract)
