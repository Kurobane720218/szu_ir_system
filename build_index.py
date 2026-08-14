"""Build inverted index and TF-IDF vectors.

Usage:
    python build_index.py --data data/research_results.jsonl --out index
    python build_index.py --data data/research_results.jsonl --out index --use-llm-tokenizer
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm

from tokenizer import normalize_text, tokenize


FIELD_ALIASES = {
    "title": ["title", "paper_title", "成果名称", "题目"],
    "authors": ["authors", "author", "作者"],
    "venue": ["venue", "journal", "conference", "source", "会议", "期刊"],
    "institute": ["institute", "lab", "research_institute", "研究所", "中心"],
    "year": ["year", "publication_year", "pub_year", "date", "发表年份"],
    "link": ["link", "url", "href", "detail_url", "详情链接"],
    "abstract": ["abstract", "detail_abstract", "摘要"],
    "detail_text": ["detail_text", "text", "content", "body", "正文", "text_for_index"],
}


def get_first(item: Dict[str, Any], names: Iterable[str], default: Any = "") -> Any:
    for name in names:
        if name in item and item[name] not in (None, ""):
            return item[name]
    return default


def parse_year(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    import re

    m = re.search(r"(19|20)\d{2}", text)
    return m.group(0) if m else ""


def normalize_doc(raw: Dict[str, Any], idx: int) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """Normalize one raw JSON object into doc_id, fields, and display metadata."""
    doc_id = normalize_text(raw.get("doc_id") or raw.get("id") or f"doc_{idx:06d}")
    title = normalize_text(get_first(raw, FIELD_ALIASES["title"]))
    authors = normalize_text(get_first(raw, FIELD_ALIASES["authors"]))
    venue = normalize_text(get_first(raw, FIELD_ALIASES["venue"]))
    institute = normalize_text(get_first(raw, FIELD_ALIASES["institute"]))
    year = parse_year(get_first(raw, FIELD_ALIASES["year"]))
    link = normalize_text(get_first(raw, FIELD_ALIASES["link"]))
    abstract = normalize_text(get_first(raw, FIELD_ALIASES["abstract"]))
    detail_text = normalize_text(get_first(raw, FIELD_ALIASES["detail_text"]))

    # If text_for_index exists, include it, but do not repeat too aggressively.
    text_for_index = normalize_text(raw.get("text_for_index", ""))
    body = " ".join(x for x in [abstract, detail_text, text_for_index] if x)

    fields = {
        "title": title,
        "authors": authors,
        "venue": venue,
        "institute": institute,
        "year": year,
        "body": body,
    }
    meta = {
        "doc_id": doc_id,
        "title": title or f"Untitled {idx}",
        "authors": authors,
        "venue": venue,
        "institute": institute,
        "year": year,
        "publication_year": year,
        "link": link,
        "abstract": abstract,
        "detail_text": detail_text,
        "snippet_source": body or " ".join([title, authors, venue, institute]),
    }
    return doc_id, fields, meta


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
    return docs


# ---------- Storage optimization helpers: gap encoding + Variable Byte encoding ----------

def gap_encode(numbers: List[int]) -> List[int]:
    """Encode increasing integers as gaps: [3, 10, 25] -> [3, 7, 15]."""
    gaps: List[int] = []
    prev = 0
    for n in sorted(numbers):
        gaps.append(n - prev)
        prev = n
    return gaps


def vb_encode_number(n: int) -> bytes:
    """Variable Byte encode one non-negative integer.

    The lower 7 bits of each byte store data. The highest bit marks the last byte.
    Small numbers usually occupy one byte.
    """
    if n < 0:
        raise ValueError("VB encoding requires non-negative integers.")
    chunks = [n % 128]
    n //= 128
    while n:
        chunks.insert(0, n % 128)
        n //= 128
    chunks[-1] += 128
    return bytes(chunks)


def vb_encode_list(numbers: Iterable[int]) -> bytes:
    output = bytearray()
    for n in numbers:
        output.extend(vb_encode_number(int(n)))
    return bytes(output)


def build_compressed_docid_postings(postings: Dict[str, Dict[str, Any]], docid_to_int: Dict[str, int]) -> Dict[str, bytes]:
    """Compress only docID lists for each term.

    Full postings still remain in postings.pkl for easy teaching/demo/debugging.
    compressed_postings_docids.pkl is used to demonstrate storage optimization.
    """
    compressed: Dict[str, bytes] = {}
    for term, doc_map in postings.items():
        doc_ints = [docid_to_int[doc_id] for doc_id in doc_map.keys() if doc_id in docid_to_int]
        compressed[term] = vb_encode_list(gap_encode(doc_ints))
    return compressed


# ---------- Main index building ----------

def convert_nested_defaultdict(obj: Any) -> Any:
    if isinstance(obj, defaultdict):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: convert_nested_defaultdict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_nested_defaultdict(v) for v in obj]
    return obj


def build_index(data_path: Path, out_dir: Path, use_llm_tokenizer: bool = False) -> None:
    raw_docs = read_jsonl(data_path)
    if not raw_docs:
        raise ValueError(f"No documents found in {data_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # term -> doc_id -> {tf, positions, fields}
    postings: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    positional_index: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
    field_index: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    docs_meta: Dict[str, Dict[str, str]] = {}
    doc_term_tf: Dict[str, Counter] = {}
    collection_tf: Counter = Counter()

    for idx, raw in enumerate(tqdm(raw_docs, desc="Building inverted index"), 1):
        doc_id, fields, meta = normalize_doc(raw, idx)
        docs_meta[doc_id] = meta

        term_tf: Counter = Counter()
        term_positions: Dict[str, List[int]] = defaultdict(list)
        term_field_tf: Dict[str, Counter] = defaultdict(Counter)
        position = 0

        # Field order gives stable positions across title/body/etc.
        for field_name in ["title", "authors", "venue", "institute", "year", "body"]:
            terms = tokenize(fields.get(field_name, ""), use_llm=use_llm_tokenizer)
            for term in terms:
                term_tf[term] += 1
                term_field_tf[term][field_name] += 1
                term_positions[term].append(position)
                position += 1

        doc_term_tf[doc_id] = term_tf
        collection_tf.update(term_tf)

        for term, tf in term_tf.items():
            record = {
                "tf": int(tf),
                "positions": term_positions[term],
                "fields": dict(term_field_tf[term]),
            }
            postings[term][doc_id] = record
            positional_index[term][doc_id] = term_positions[term]
            for field_name, f_tf in term_field_tf[term].items():
                field_index[field_name][term][doc_id] = int(f_tf)

    postings = convert_nested_defaultdict(postings)
    positional_index = convert_nested_defaultdict(positional_index)
    field_index = convert_nested_defaultdict(field_index)

    total_docs = len(docs_meta)
    dictionary: Dict[str, Dict[str, float]] = {}
    idf: Dict[str, float] = {}
    for term, doc_map in postings.items():
        df = len(doc_map)
        cf = collection_tf[term]
        idf_value = math.log((total_docs + 1) / (df + 1)) + 1
        dictionary[term] = {"df": df, "cf": int(cf), "idf": idf_value}
        idf[term] = idf_value

    tfidf_vectors: Dict[str, Dict[str, float]] = {}
    doc_norms: Dict[str, float] = {}
    for doc_id, tf_counter in doc_term_tf.items():
        vector: Dict[str, float] = {}
        norm_sq = 0.0
        for term, tf in tf_counter.items():
            tf_weight = 1.0 + math.log(tf)
            weight = tf_weight * idf[term]
            vector[term] = weight
            norm_sq += weight * weight
        tfidf_vectors[doc_id] = vector
        doc_norms[doc_id] = math.sqrt(norm_sq)

    # Helper indexes for wildcard and spelling correction.
    terms_sorted = sorted(dictionary.keys())
    permuterm_index: Dict[str, List[str]] = {}
    for term in terms_sorted:
        marked = term + "$"
        for i in range(len(marked)):
            permuterm_index.setdefault(marked[i:] + marked[:i], []).append(term)

    kgram_index: Dict[str, List[str]] = defaultdict(list)
    k = 3
    for term in terms_sorted:
        padded = f"${term}$"
        grams = {padded[i : i + k] for i in range(max(len(padded) - k + 1, 1))}
        for gram in grams:
            kgram_index[gram].append(term)
    kgram_index = dict(kgram_index)

    docid_to_int = {doc_id: i for i, doc_id in enumerate(sorted(docs_meta.keys()), 1)}
    compressed_postings = build_compressed_docid_postings(postings, docid_to_int)

    # Save files.
    with (out_dir / "docs_meta.json").open("w", encoding="utf-8") as f:
        json.dump(docs_meta, f, ensure_ascii=False, indent=2)
    with (out_dir / "dictionary.json").open("w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    with (out_dir / "docid_map.json").open("w", encoding="utf-8") as f:
        json.dump(docid_to_int, f, ensure_ascii=False, indent=2)

    for filename, obj in [
        ("postings.pkl", postings),
        ("positional_index.pkl", positional_index),
        ("field_index.pkl", field_index),
        ("tfidf.pkl", tfidf_vectors),
        ("doc_norms.pkl", doc_norms),
        ("permuterm_index.pkl", permuterm_index),
        ("kgram_index.pkl", kgram_index),
        ("compressed_postings_docids.pkl", compressed_postings),
    ]:
        with (out_dir / filename).open("wb") as f:
            pickle.dump(obj, f)

    storage_report = {
        "num_docs": total_docs,
        "vocab_size": len(dictionary),
        "num_term_doc_pairs": sum(len(v) for v in postings.values()),
        "index_files_bytes": {
            p.name: p.stat().st_size for p in sorted(out_dir.glob("*")) if p.is_file()
        },
        "storage_optimization": "compressed_postings_docids.pkl stores docID lists using gap encoding + Variable Byte encoding.",
    }
    with (out_dir / "storage_report.json").open("w", encoding="utf-8") as f:
        json.dump(storage_report, f, ensure_ascii=False, indent=2)

    print("\nIndex built successfully.")
    print(f"Documents: {total_docs}")
    print(f"Vocabulary size: {len(dictionary)}")
    print(f"Term-document pairs: {storage_report['num_term_doc_pairs']}")
    print(f"Output directory: {out_dir}")
    print("Storage optimization file: compressed_postings_docids.pkl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/research_results.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("index"))
    parser.add_argument("--use-llm-tokenizer", action="store_true", help="Use LLM tokenizer when configured; otherwise fallback to jieba.")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Data file not found: {args.data}. Put your JSONL file at data/research_results.jsonl."
        )
    build_index(args.data, args.out, use_llm_tokenizer=args.use_llm_tokenizer)


if __name__ == "__main__":
    main()
