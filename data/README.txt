此目录用于存放科研成果 JSONL 数据。

为了让项目下载后可以立即测试，这里附带了一个 10 条记录的 demo research_results.jsonl。
正式提交大作业时，请用你已经爬取的 322 条深大计算机科研成果数据覆盖：

data/research_results.jsonl

然后重新运行：
python build_index.py --data data/research_results.jsonl --out index
