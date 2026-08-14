"""Streamlit front-end for SZU IR system.

Deleted unstable/non-required features: image display, voice interaction,
email subscription, and personalized recommendation. The UI focuses on core IR
features and LLM summary.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from config import ENABLE_LLM_SUMMARY, INDEX_DIR, RANK_ALPHA, RANK_BETA, TOP_K
from llm_helper import summarize_results
from search_engine import SearchEngine, highlight_terms


st.set_page_config(page_title="深大计算机科研成果信息检索系统", page_icon="🔎", layout="wide")

st.markdown(
    """
<style>
.main .block-container {padding-top: 2rem; max-width: 1180px;}
.result-card {border: 1px solid #E5E7EB; border-radius: 12px; padding: 18px 20px; margin: 12px 0; background: #FFFFFF;}
.result-title {font-size: 1.12rem; font-weight: 650; margin-bottom: 6px;}
.meta {color: #4B5563; font-size: 0.92rem; line-height: 1.55;}
.score {font-family: monospace; background: #F3F4F6; padding: 2px 6px; border-radius: 6px;}
mark {background: #FDE68A; color: #111827; padding: 0 2px; border-radius: 3px;}
.small-note {color: #6B7280; font-size: 0.9rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_engine() -> SearchEngine:
    return SearchEngine(INDEX_DIR)


st.title("结合大模型的深大计算机科研成果信息检索系统")
st.caption("倒排索引 + TF-IDF + 余弦相似度 + 时效性排序 + 可选大模型结果总结")

with st.sidebar:
    st.header("检索设置")
    field = st.selectbox("检索域", ["全部", "题目", "作者", "摘要/正文", "会议/期刊", "研究所", "年份"], index=0)
    top_k = st.slider("返回结果数量", min_value=1, max_value=20, value=TOP_K)
    use_recency = st.checkbox("启用时效性排序", value=True)
    use_llm_tokenizer = st.checkbox("查询阶段使用大模型分词", value=os.getenv("USE_LLM_TOKENIZER", "false").lower() == "true")
    st.divider()
    st.markdown("**当前排序公式**")
    if use_recency:
        st.code(f"final_score = {RANK_ALPHA:.2f} * cosine + {RANK_BETA:.2f} * recency")
    else:
        st.code("final_score = cosine")
    st.markdown("**示例查询**")
    st.write("医学图像分割")
    st.write("semantic segmentation")
    st.write("federated learning")
    st.write("title:Geo-localization")
    st.write("/near semantic segmentation 5")
    st.write("geo*")

query = st.text_input(
    "请输入查询内容",
    placeholder="例如：医学图像分割 / semantic segmentation / title:Geo-localization / /near semantic segmentation 5",
)

col1, col2 = st.columns([1, 5])
with col1:
    do_search = st.button("检索", type="primary", use_container_width=True)
with col2:
    st.markdown(
        "<span class='small-note'>本版本已删除图片展示、语音交互、邮件订阅和个性化推荐，聚焦信息检索核心功能。</span>",
        unsafe_allow_html=True,
    )

if do_search and query.strip():
    try:
        engine = load_engine()
    except Exception as e:
        st.error(f"索引加载失败：{e}")
        st.info("请先运行：python build_index.py --data data/research_results.jsonl --out index")
        st.stop()

    with st.spinner("正在检索..."):
        results, info = engine.search(
            query=query,
            field=field,
            top_k=top_k,
            use_recency=use_recency,
            use_llm_tokenizer=use_llm_tokenizer,
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("查询耗时", f"{info['elapsed_ms']:.2f} ms")
    m2.metric("候选文档数", info.get("candidate_count", 0))
    m3.metric("返回结果数", len(results))
    m4.metric("检索域", info.get("field", "all"))

    with st.expander("查看 Query 分词和诊断信息", expanded=True):
        st.write("原始 Query：", info.get("original_query"))
        st.write("实际 Query：", info.get("effective_query"))
        st.write("Query Terms：", " / ".join(info.get("query_terms", [])) or "无")
        if info.get("expanded_terms"):
            st.write("通配符展开 Terms：", " / ".join(info.get("expanded_terms", [])))
        st.write("排序公式：", info.get("ranking_formula"))
        suggestions = info.get("suggestions", {})
        if suggestions:
            for wrong, cands in suggestions.items():
                st.warning(f"你是不是想搜：{wrong} → {' / '.join(cands)}")
        if info.get("near"):
            st.write("邻近搜索参数：", info["near"])

    if ENABLE_LLM_SUMMARY and results:
        with st.spinner("正在生成大模型总结..."):
            summary = summarize_results(query, results)
        if summary:
            st.subheader("大模型对 Top 结果的总结")
            st.info(summary)

    st.subheader("倒排索引返回的检索结果")
    if not results:
        st.warning("没有找到匹配结果。可以尝试更换关键词，或查看纠错建议。")
    else:
        q_terms = info.get("query_terms", [])
        for idx, item in enumerate(results, 1):
            title = item.get("title") or "Untitled"
            link = item.get("link") or ""
            title_html = highlight_terms(title, q_terms)
            snippet_html = highlight_terms(item.get("snippet", ""), q_terms)
            if link:
                title_block = f"<a href='{link}' target='_blank'>{title_html}</a>"
            else:
                title_block = title_html

            st.markdown(
                f"""
<div class="result-card">
  <div class="result-title">{idx}. {title_block}</div>
  <div class="meta">
    作者：{item.get('authors','') or '未知'}<br/>
    来源：{item.get('venue','') or '未知'} | 年份：{item.get('year','') or '未知'} | 研究所：{item.get('institute','') or '未知'}
  </div>
  <p>{snippet_html}</p>
  <div>
    综合分值：<span class="score">{item.get('score',0):.4f}</span>
    &nbsp; 余弦相似度：<span class="score">{item.get('cosine_score',0):.4f}</span>
    &nbsp; 时效性分值：<span class="score">{item.get('recency_score',0):.4f}</span>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
else:
    st.info("请输入 query 后点击“检索”。首次运行前请先构建索引。")
