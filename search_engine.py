"""Search engine core: inverted index + TF-IDF + cosine + recency ranking."""

from __future__ import annotations

import html
import json
import math
import pickle
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from config import CURRENT_YEAR, INDEX_DIR, MAX_WILDCARD_EXPANSIONS, RANK_ALPHA, RANK_BETA, TOP_K
from spell_correct import suggest_terms
from tokenizer import normalize_text, tokenize
from wildcard import expand_wildcard

FIELD_MAP = {
    "全部": None,
    "all": None,
    "题目": "title",
    "title": "title",
    "标题": "title",
    "作者": "authors",
    "authors": "authors",
    "author": "authors",
    "会议/期刊": "venue",
    "venue": "venue",
    "来源": "venue",
    "研究所": "institute",
    "institute": "institute",
    "摘要/正文": "body",
    "正文": "body",
    "body": "body",
    "年份": "year",
    "year": "year",
}

PREFIX_TO_FIELD = {
    "title": "title",
    "题目": "title",
    "标题": "title",
    "author": "authors",
    "authors": "authors",
    "作者": "authors",
    "venue": "venue",
    "会议": "venue",
    "期刊": "venue",
    "institute": "institute",
    "研究所": "institute",
    "body": "body",
    "正文": "body",
    "year": "year",
    "年份": "year",
}


class SearchEngine:
    """Load local index files and answer queries."""

    def __init__(self, index_dir: str | Path = INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.docs_meta: Dict[str, Dict[str, Any]] = self._load_json("docs_meta.json")
        self.dictionary: Dict[str, Dict[str, float]] = self._load_json("dictionary.json")
        self.postings: Dict[str, Dict[str, Dict[str, Any]]] = self._load_pickle("postings.pkl")
        self.positional_index: Dict[str, Dict[str, List[int]]] = self._load_pickle("positional_index.pkl")
        self.field_index: Dict[str, Dict[str, Dict[str, int]]] = self._load_pickle("field_index.pkl")
        self.tfidf_vectors: Dict[str, Dict[str, float]] = self._load_pickle("tfidf.pkl")
        self.doc_norms: Dict[str, float] = self._load_pickle("doc_norms.pkl")
        self.vocabulary = set(self.dictionary.keys())
        self.total_docs = len(self.docs_meta)
        self.alpha = RANK_ALPHA
        self.beta = RANK_BETA
        self.current_year = CURRENT_YEAR

    def _load_json(self, filename: str) -> Dict[str, Any]:
        path = self.index_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing index file: {path}. Run build_index.py first.")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_pickle(self, filename: str) -> Any:
        path = self.index_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing index file: {path}. Run build_index.py first.")
        with path.open("rb") as f:
            return pickle.load(f)

    # ---------------- Ranking functions ----------------
    def _idf(self, term: str) -> float:
        item = self.dictionary.get(term)
        if item:
            return float(item.get("idf", 1.0))
        # Smooth idf for unseen query terms.
        return math.log((self.total_docs + 1) / 1) + 1

    def _recency_score(self, year: Any) -> float:
        """Calculate document recency score in [0, 1].

        recency(d) = 1 / (1 + current_year - year(d))

        Missing or invalid years get a small default score. Future years are
        clipped to 1.0 to avoid unreasonable boosts.
        """
        try:
            match = re.search(r"(19|20)\d{2}", str(year))
            if not match:
                return 0.10
            y = int(match.group(0))
            if y >= self.current_year:
                return 1.0
            if y < 1900:
                return 0.10
            return 1.0 / (1.0 + self.current_year - y)
        except Exception:
            return 0.10

    def _query_vector(self, q_terms: Sequence[str]) -> Tuple[Dict[str, float], float]:
        q_tf = Counter(q_terms)
        q_vec: Dict[str, float] = {}
        norm_sq = 0.0
        for term, tf in q_tf.items():
            tf_weight = 1.0 + math.log(tf)
            weight = tf_weight * self._idf(term)
            q_vec[term] = weight
            norm_sq += weight * weight
        return q_vec, math.sqrt(norm_sq)

    def _candidate_docs(self, q_terms: Sequence[str], field: Optional[str] = None) -> Set[str]:
        candidates: Set[str] = set()
        for term in q_terms:
            if field:
                candidates.update(self.field_index.get(field, {}).get(term, {}).keys())
            else:
                candidates.update(self.postings.get(term, {}).keys())
        return candidates

    def _rank_candidates(
        self,
        q_terms: Sequence[str],
        candidate_docs: Iterable[str],
        top_k: int,
        use_recency: bool = True,
    ) -> List[Dict[str, Any]]:
        """Rank candidate docs by cosine similarity plus recency.

        final_score = alpha * cosine_score + beta * recency_score
        If use_recency=False, final_score is pure cosine_score.
        """
        q_vec, q_norm = self._query_vector(q_terms)
        if q_norm == 0:
            return []

        results: List[Dict[str, Any]] = []
        for doc_id in candidate_docs:
            doc_vec = self.tfidf_vectors.get(doc_id, {})
            doc_norm = self.doc_norms.get(doc_id, 0.0)
            if doc_norm == 0:
                continue

            # Cosine numerator: dot product between query vector and document vector.
            dot = 0.0
            for term, q_weight in q_vec.items():
                d_weight = doc_vec.get(term, 0.0)
                dot += q_weight * d_weight

            cosine_score = dot / (q_norm * doc_norm)
            if cosine_score <= 0:
                continue

            meta = self.docs_meta.get(doc_id, {})
            recency = self._recency_score(meta.get("year") or meta.get("publication_year"))
            if use_recency:
                final_score = self.alpha * cosine_score + self.beta * recency
            else:
                final_score = cosine_score

            item = {
                "doc_id": doc_id,
                "score": final_score,
                "cosine_score": cosine_score,
                "recency_score": recency,
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "venue": meta.get("venue", ""),
                "year": meta.get("year") or meta.get("publication_year", ""),
                "institute": meta.get("institute", ""),
                "link": meta.get("link", ""),
                "abstract": meta.get("abstract", ""),
                "snippet": self._make_snippet(meta, q_terms),
            }
            results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ---------------- Query parsing and search modes ----------------
    def _parse_field_prefix(self, query: str, selected_field: str | None) -> Tuple[str, Optional[str]]:
        field = FIELD_MAP.get(selected_field or "全部", None)
        m = re.match(r"^\s*([A-Za-z\u4e00-\u9fff/]+)\s*[:：]\s*(.+)$", query)
        if m:
            prefix = m.group(1).lower()
            rest = m.group(2).strip()
            if prefix in PREFIX_TO_FIELD:
                return rest, PREFIX_TO_FIELD[prefix]
        return query, field

    def _parse_near(self, query: str) -> Optional[Tuple[str, str, int]]:
        # Format: /near semantic segmentation 5
        parts = query.strip().split()
        if len(parts) >= 4 and parts[0].lower() == "/near":
            try:
                k = int(parts[-1])
                term1 = parts[1].lower()
                term2 = parts[2].lower()
                return term1, term2, k
            except Exception:
                return None
        return None

    def _near_candidates(self, term1: str, term2: str, k: int) -> Set[str]:
        p1 = self.positional_index.get(term1, {})
        p2 = self.positional_index.get(term2, {})
        docs = set(p1.keys()) & set(p2.keys())
        matched: Set[str] = set()
        for doc_id in docs:
            positions1 = p1.get(doc_id, [])
            positions2 = p2.get(doc_id, [])
            i = j = 0
            while i < len(positions1) and j < len(positions2):
                diff = positions1[i] - positions2[j]
                if abs(diff) <= k:
                    matched.add(doc_id)
                    break
                if diff < 0:
                    i += 1
                else:
                    j += 1
        return matched

    def _expand_query_terms(self, query: str, use_llm: bool | None = None) -> Tuple[List[str], List[str]]:
        """Tokenize query and expand wildcard terms.

        Returns: (query_terms_for_ranking, wildcard_expanded_terms)
        """
        raw_parts = query.strip().split()
        expanded_terms: List[str] = []
        if any("*" in part for part in raw_parts):
            for part in raw_parts:
                if "*" in part:
                    expanded_terms.extend(
                        expand_wildcard(part.lower(), self.vocabulary, MAX_WILDCARD_EXPANSIONS)
                    )
                else:
                    expanded_terms.extend(tokenize(part, use_llm=use_llm))
            return expanded_terms, expanded_terms
        q_terms = tokenize(query, use_llm=use_llm)
        return q_terms, []

    def search(
        self,
        query: str,
        field: str | None = "全部",
        top_k: int = TOP_K,
        use_recency: bool = True,
        use_llm_tokenizer: bool | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Search and return results plus diagnostics."""
        start = time.perf_counter()
        original_query = query
        query = normalize_text(query)
        query, field_key = self._parse_field_prefix(query, field)

        diagnostics: Dict[str, Any] = {
            "original_query": original_query,
            "effective_query": query,
            "field": field_key or "all",
            "query_terms": [],
            "expanded_terms": [],
            "candidate_count": 0,
            "elapsed_ms": 0.0,
            "suggestions": {},
            "ranking_formula": "final_score = 0.85*cosine_score + 0.15*recency_score" if use_recency else "final_score = cosine_score",
        }

        if not query:
            return [], diagnostics

        near = self._parse_near(query)
        if near:
            term1, term2, k = near
            q_terms = [term1, term2]
            candidates = self._near_candidates(term1, term2, k)
            diagnostics.update({"query_terms": q_terms, "candidate_count": len(candidates), "near": {"term1": term1, "term2": term2, "k": k}})
            results = self._rank_candidates(q_terms, candidates, top_k=top_k, use_recency=use_recency)
            diagnostics["elapsed_ms"] = (time.perf_counter() - start) * 1000
            return results, diagnostics

        q_terms, expanded_terms = self._expand_query_terms(query, use_llm=use_llm_tokenizer)
        q_terms = [t for t in q_terms if t]
        diagnostics["query_terms"] = q_terms
        diagnostics["expanded_terms"] = expanded_terms

        # Suggestions for terms not found in vocabulary.
        suggestions: Dict[str, List[str]] = {}
        for term in q_terms:
            if term not in self.vocabulary and " " not in term and "*" not in term:
                cand = suggest_terms(term, self.vocabulary, top_k=3)
                if cand:
                    suggestions[term] = cand
        diagnostics["suggestions"] = suggestions

        candidates = self._candidate_docs(q_terms, field_key)
        diagnostics["candidate_count"] = len(candidates)
        results = self._rank_candidates(q_terms, candidates, top_k=top_k, use_recency=use_recency)
        diagnostics["elapsed_ms"] = (time.perf_counter() - start) * 1000
        return results, diagnostics

    # ---------------- Display helpers ----------------
    def _make_snippet(self, meta: Dict[str, Any], q_terms: Sequence[str], max_chars: int = 260) -> str:
        text = normalize_text(meta.get("abstract") or meta.get("snippet_source") or meta.get("detail_text") or meta.get("title"))
        if not text:
            return ""
        lower = text.lower()
        pos = -1
        for term in q_terms:
            if not term:
                continue
            p = lower.find(term.lower())
            if p >= 0:
                pos = p
                break
        if pos < 0:
            return text[:max_chars] + ("..." if len(text) > max_chars else "")
        start = max(pos - max_chars // 3, 0)
        end = min(start + max_chars, len(text))
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."
        return snippet


def highlight_terms(text: str, terms: Sequence[str]) -> str:
    """HTML-escape text and mark query terms."""
    escaped = html.escape(text or "")
    # Highlight longer terms first to avoid fragmenting phrases.
    for term in sorted({t for t in terms if t}, key=len, reverse=True):
        safe_term = html.escape(term)
        if not safe_term:
            continue
        pattern = re.compile(re.escape(safe_term), flags=re.IGNORECASE)
        escaped = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)
    return escaped
