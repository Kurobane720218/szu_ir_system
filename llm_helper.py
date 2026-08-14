"""Optional LLM summary for Top-K retrieval results."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


def summarize_results(query: str, results: List[Dict[str, Any]]) -> str:
    """Summarize Top-K results by LLM. Return empty string if disabled/unavailable."""
    if os.getenv("ENABLE_LLM_SUMMARY", "false").lower() != "true":
        return ""
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    model = os.getenv("LLM_MODEL", "")
    if not api_key or not model or not results:
        return ""

    try:
        from openai import OpenAI

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        lines = []
        for i, item in enumerate(results[:10], 1):
            lines.append(
                f"{i}. 标题：{item.get('title','')}\n"
                f"   作者：{item.get('authors','')}\n"
                f"   来源：{item.get('venue','')}\n"
                f"   年份：{item.get('year','')}\n"
                f"   综合分值：{item.get('score',0):.4f}\n"
                f"   摘要片段：{item.get('snippet','')[:350]}"
            )
        prompt = f"""
用户查询：{query}

下面是信息检索系统基于倒排索引、TF-IDF、余弦相似度和时效性排序返回的 Top 10 结果：
{chr(10).join(lines)}

请用中文生成一段简洁总结，要求：
1. 概括这些结果主要集中在哪些研究方向；
2. 指出最相关或最新的代表性成果；
3. 不要编造结果中没有的信息；
4. 控制在 150 字以内。
""".strip()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是科研成果检索系统的结果总结助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"大模型总结暂不可用：{e}"
