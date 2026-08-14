"""Simple spelling correction for retrieval terms."""

from __future__ import annotations

from typing import Iterable, List, Tuple


def edit_distance(a: str, b: str, max_distance: int = 2) -> int:
    """Levenshtein distance with a soft early cut-off."""
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = cur[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(val)
            row_min = min(row_min, val)
        if row_min > max_distance:
            return max_distance + 1
        prev = cur
    return prev[-1]


def suggest_terms(term: str, vocabulary: Iterable[str], top_k: int = 3, max_distance: int = 2) -> List[str]:
    term = term.lower().strip()
    if not term:
        return []
    candidates: List[Tuple[int, int, str]] = []
    for vocab_term in vocabulary:
        # Keep correction cheap: mostly useful for English terms.
        if abs(len(vocab_term) - len(term)) > max_distance:
            continue
        dist = edit_distance(term, vocab_term, max_distance=max_distance)
        if dist <= max_distance:
            candidates.append((dist, len(vocab_term), vocab_term))
    candidates.sort()
    return [x[2] for x in candidates[:top_k]]
