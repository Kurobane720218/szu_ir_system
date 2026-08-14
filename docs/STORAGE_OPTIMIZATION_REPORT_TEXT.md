# 报告可用文字：存储优化

在存储优化方面，系统首先将 dictionary、postings list、文档元数据和 TF-IDF 向量分开存储。`dictionary.json` 只保存唯一词项，避免在不同文档中重复存储词项；`postings.pkl` 只保存词项到文档的映射关系，包括 docID、词频、位置信息和字段信息；`docs_meta.json` 单独保存标题、作者、年份和链接等展示用元数据。这样的设计避免了在倒排记录表中重复存储完整文档内容和展示信息。

进一步地，系统对 postings list 中的 docID 列表进行了 gap encoding 和 Variable Byte Encoding 压缩。由于每个词项对应的 docID 列表可以按升序排列，因此可以将原始 docID 序列转换为相邻 docID 的差值序列。例如，原始 docID 序列 `[3, 10, 25, 40]` 可以转换为 gap 序列 `[3, 7, 15, 15]`。gap 通常比原始 docID 小，更适合使用变长字节编码存储。Variable Byte Encoding 使用 7 bit 存储数据，最高位作为结束标记，小整数通常只需要 1 个字节，因此可以减少 postings list 的实际存储空间。

未压缩情况下，倒排索引的空间复杂度可以表示为：

\[
O(|V|+P+L)
\]

其中，\(|V|\) 为词典大小，\(P\) 为所有非零 term-document 对数量，\(L\) 为所有词项出现位置的总数量。使用 gap encoding 和 Variable Byte Encoding 后，理论上的大 O 空间复杂度仍然为 \(O(P)\)，但 docID 部分的实际存储位数由 \(O(P\log N)\) 降低为 \(O(\sum VB(gap_i))\)。因此，该优化主要降低常数空间开销，提高索引文件的实际存储效率。

代码实现位于 `build_index.py` 文件中，主要包括 `gap_encode()`、`vb_encode_number()`、`vb_encode_list()` 和 `build_compressed_docid_postings()` 函数。构建索引后，系统会生成 `compressed_postings_docids.pkl` 和 `storage_report.json`，可用于截图展示压缩后的 docID postings 文件和原始 postings 文件的大小对比。
