# 结合大模型的深大计算机科研成果信息检索系统（含时效性排序版）

- 中英文 query 分词，支持大模型 API 分词和 jieba fallback。
- 倒排索引 inverted index。
- TF-IDF 文档向量。
- 余弦相似度相关性计算。
- 时效性排序：final_score = 0.85 × cosine_score + 0.15 × recency_score。
- Top 10 结果、查询耗时、相关性分值展示。
- 标题超链接和关键词高亮。
- 可选大模型总结 Top 10 检索结果。
- 可选：按域检索、邻近搜索、通配符查询、查询纠错。
- 存储优化演示：docID gap encoding + Variable Byte encoding，生成 `compressed_postings_docids.pkl`。

## 1. 安装运行

```powershell
cd szu_ir_system_recency
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

把数据放到：

```text
data/research_results.jsonl
```

构建索引：

```powershell
python build_index.py --data data\research_results.jsonl --out index
```

验证系统：

```powershell
python verify_system.py
```

启动前端：

```powershell
streamlit run app.py
```

## 2. 数据格式

`research_results.jsonl` 每行是一篇文档，例如：

```json
{"title":"A Weakly Supervised Semantic Segmentation Method", "authors":"A; B", "venue":"CVPR", "publication_year":"2025", "institute":"Computer Vision Institute", "link":"https://example.com", "detail_abstract":"...", "detail_text":"..."}
```

字段名不完全一致也可以，系统会尽量读取：

- `title`
- `authors` / `author`
- `venue` / `journal` / `conference`
- `institute` / `lab`
- `publication_year` / `year`
- `link` / `url`
- `detail_abstract` / `abstract`
- `detail_text` / `text` / `text_for_index`

## 3. 大模型配置

复制配置：

```powershell
copy .env.example .env
```

打开 `.env`：

```text
USE_LLM_TOKENIZER=false
ENABLE_LLM_SUMMARY=false
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
RANK_ALPHA=0.85
RANK_BETA=0.15
CURRENT_YEAR=2026
```

如果启用大模型分词：

```text
USE_LLM_TOKENIZER=true
```

然后重新构建索引：

```powershell
python build_index.py --data data\research_results.jsonl --out index --use-llm-tokenizer
```

如果启用大模型总结：

```text
ENABLE_LLM_SUMMARY=true
```

然后重启 Streamlit。

## 4. 时效性排序实现位置

在 `search_engine.py` 中：

```python
def _recency_score(self, year):
    return 1.0 / (1.0 + self.current_year - y)
```

核心排序逻辑：

```python
cosine_score = dot / (q_norm * doc_norm)
recency = self._recency_score(meta.get("year"))
final_score = self.alpha * cosine_score + self.beta * recency
```

页面会显示：

```text
综合分值、余弦相似度、时效性分值
```

## 5. 存储优化实现位置

在 `build_index.py` 中：

- `gap_encode(numbers)`
- `vb_encode_number(n)`
- `vb_encode_list(numbers)`
- `build_compressed_docid_postings(postings, docid_to_int)`

构建索引后会生成：

```text
index/compressed_postings_docids.pkl
index/storage_report.json
```

对比文件大小：

```powershell
python -c "import os; print('postings.pkl:', os.path.getsize('index/postings.pkl')); print('compressed:', os.path.getsize('index/compressed_postings_docids.pkl'))"
```

注意：`compressed_postings_docids.pkl` 只压缩 docID 列表用于展示存储优化。完整的 `postings.pkl` 仍保留 `tf / positions / fields`，方便课程演示、邻近搜索和调试。

## 6. 推荐演示查询

```text
医学图像分割
semantic segmentation
federated learning
title:Geo-localization
/near semantic segmentation 5
geo*
segmantation
```
