"""Tokenization utilities.

The system prefers LLM tokenization when USE_LLM_TOKENIZER=true and a compatible
OpenAI-style API is configured. If the API fails, it falls back to jieba + simple
English tokenization, so the whole IR system remains runnable offline.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Iterable, List

try:
    import jieba  # type: ignore
except Exception:  # fallback when jieba is not installed yet
    jieba = None
from dotenv import load_dotenv

load_dotenv()

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with", "by",
    "from", "is", "are", "was", "were", "be", "been", "this", "that", "these", "those",
    "研究", "一种", "基于", "方法", "系统", "及其", "进行", "实现", "中的", "一个", "一种",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*|\d{2,4}|[\u4e00-\u9fff]+")


def normalize_text(text: object) -> str:
    """Convert arbitrary input to a normalized string."""
    if text is None:
        return ""
    if isinstance(text, (list, tuple, set)):
        text = " ".join(str(x) for x in text)
    text = str(text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_terms(terms: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for term in terms:
        t = str(term).strip().lower()
        t = t.strip("`'\".,;:!?()[]{}<>《》“”‘’|/\\")
        if not t:
            continue
        if t in STOPWORDS:
            continue
        # Remove pure punctuation tokens but keep Chinese/English/digits.
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", t):
            continue
        # Avoid one-character English tokens. Keep Chinese single chars only when useful.
        if len(t) == 1 and re.fullmatch(r"[a-z]", t):
            continue
        if t not in seen:
            cleaned.append(t)
            seen.add(t)
    return cleaned


def local_tokenize(text: object) -> List[str]:
    """Fallback tokenizer: jieba for Chinese and regex for English/digits."""
    text = normalize_text(text)
    if not text:
        return []

    # jieba can segment mixed Chinese/English text; if unavailable, regex still works.
    if jieba is not None:
        jieba_terms = list(jieba.cut(text, cut_all=False))
    else:
        jieba_terms = []
        # Crude Chinese fallback: add contiguous Chinese chunks plus 2-gram/3-gram terms.
        for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
            jieba_terms.append(chunk)
            jieba_terms.extend(chunk[i:i+2] for i in range(max(len(chunk)-1, 0)))
            jieba_terms.extend(chunk[i:i+3] for i in range(max(len(chunk)-2, 0)))
    regex_terms = TOKEN_RE.findall(text)
    terms = jieba_terms + regex_terms

    # Add simple English phrase terms for common academic phrases.
    lower_text = text.lower()
    phrase_candidates = [
        "semantic segmentation",
        "federated learning",
        "weakly supervised",
        "medical image",
        "image segmentation",
        "geo-localization",
        "cross-view",
        "recommendation system",
        "computer vision",
        "deep learning",
        "machine learning",
    ]
    for phrase in phrase_candidates:
        if phrase in lower_text:
            terms.append(phrase)
    return _clean_terms(terms)


@lru_cache(maxsize=2048)
def llm_tokenize_cached(text: str) -> tuple[str, ...]:
    """Tokenize text by LLM. Raises an exception if API is unavailable."""
    from openai import OpenAI

    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    model = os.getenv("LLM_MODEL", "")
    if not api_key or not model:
        raise RuntimeError("LLM tokenizer is not configured.")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    prompt = f"""
你是信息检索系统的分词器。请把下面的中文、英文或中英混合科研文本转换成适合倒排索引检索的 terms。
要求：
1. 保留重要中文学术短语，例如“医学图像分割”“跨视角地理定位”。
2. 保留重要英文短语和单词，例如“semantic segmentation”“federated learning”。
3. 去掉停用词和无意义符号。
4. 只输出 JSON 数组，例如 ["semantic segmentation", "semantic", "segmentation"]。

文本：{text}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise tokenizer for an information retrieval system."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or "[]"
    match = re.search(r"\[[\s\S]*\]", content)
    if match:
        content = match.group(0)
    terms = json.loads(content)
    if not isinstance(terms, list):
        raise ValueError("LLM did not return a JSON list.")
    return tuple(_clean_terms(str(x) for x in terms))


def tokenize(text: object, use_llm: bool | None = None) -> List[str]:
    """Tokenize text into retrieval terms.

    Parameters
    ----------
    text:
        Input string, list, or any object convertible to string.
    use_llm:
        If True, try LLM tokenization first. If None, read USE_LLM_TOKENIZER
        from environment.
    """
    text = normalize_text(text)
    if not text:
        return []
    if use_llm is None:
        use_llm = os.getenv("USE_LLM_TOKENIZER", "false").lower() == "true"
    if use_llm:
        try:
            return list(llm_tokenize_cached(text))
        except Exception:
            # Stable fallback for demos and offline runs.
            return local_tokenize(text)
    return local_tokenize(text)
