"""Quick verification script.

Run after building index:
    python verify_system.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from search_engine import SearchEngine


def file_size(path: str | Path) -> int:
    p = Path(path)
    return p.stat().st_size if p.exists() else 0


def main() -> None:
    index_dir = Path("index")
    engine = SearchEngine(index_dir)
    print("==== Index summary ====")
    print("文档数量:", engine.total_docs)
    print("词典大小:", len(engine.dictionary))
    print("postings 词项数量:", len(engine.postings))
    print("TF-IDF 文档向量数量:", len(engine.tfidf_vectors))

    print("\n==== Storage files ====")
    for name in [
        "dictionary.json",
        "postings.pkl",
        "compressed_postings_docids.pkl",
        "tfidf.pkl",
        "doc_norms.pkl",
        "docs_meta.json",
    ]:
        print(f"{name}: {file_size(index_dir / name)} bytes")

    report_path = index_dir / "storage_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print("term-document pairs:", report.get("num_term_doc_pairs"))

    print("\n==== Recency score examples ====")
    for y in ["2026", "2025", "2024", "2023", "2020", ""]:
        print(f"year={y or 'missing'} -> recency={engine._recency_score(y):.4f}")

    print("\n==== Search examples ====")
    for q in ["医学图像分割", "semantic segmentation", "federated learning", "geo*", "/near semantic segmentation 5"]:
        results, info = engine.search(q, top_k=5, use_recency=True)
        print(f"\nQuery: {q}")
        print(f"elapsed_ms={info['elapsed_ms']:.2f}, candidates={info['candidate_count']}, terms={info['query_terms']}")
        if info.get("suggestions"):
            print("suggestions:", info["suggestions"])
        for i, r in enumerate(results[:3], 1):
            print(
                f"{i}. score={r['score']:.4f}, cosine={r['cosine_score']:.4f}, "
                f"recency={r['recency_score']:.4f}, year={r.get('year','')}, title={r.get('title','')[:80]}"
            )


if __name__ == "__main__":
    main()
