#https://github.com/impresso/CLEF-HIPE-2020-internal/blob/master/lib/helpers/__init__.py

import re

from typing import NamedTuple
from stringdist import levenshtein_norm

HYPHENS = ["-", "¬"]
PATTERN_HYPHEN_CLEANING = re.compile(fr'[{"".join(HYPHENS)}]\s*')


ImpressoDocument = NamedTuple(
    "ImpressoDocument",
    [
        ("newspaper", str),
        ("date", str),
        ("id", str),
        ("filename", str),
        ("filepath", str),
        ("segments", dict),
        ("autosentences", dict),
        ("mentions", dict),
        ("links", list),
        ("relations", list),
        ("text", str),
    ],
)


def compute_levenshtein_distance(surface: str, transcript: str) -> int:
    """Compute the normalized Levensthein distance between two strings after cleaning
    :param str surface: a reference string.
    :param str transcript: a candidate string.
    :return: Levensthein distance
    :rtype: int
    """

    def clean(text: str) -> str:
        """
        Remove the symbols "-" or "¬" together with potential whitespace which may follow
        """
        return PATTERN_HYPHEN_CLEANING.sub("", text,)

    return levenshtein_norm(clean(surface), clean(transcript))