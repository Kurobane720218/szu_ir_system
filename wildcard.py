"""Wildcard query expansion."""

from __future__ import annotations

import re
from typing import Iterable, List


def expand_wildcard(pattern: str, vocabulary: Iterable[str], max_terms: int = 80) -> List[str]:
    """Expand a wildcard pattern like geo*, *learning, cross*view.

    This implementation scans the vocabulary, which is simple and reliable for a
    course project. The project also builds a permuterm index for demonstration.
    """
    pattern = pattern.strip().lower()
    if "*" not in pattern:
        return [pattern] if pattern else []
    regex = "^" + re.escape(pattern).replace("\\*", ".*") + "$"
    compiled = re.compile(regex)
    matches = [term for term in vocabulary if compiled.match(term)]
    matches.sort(key=lambda t: (len(t), t))
    return matches[:max_terms]
