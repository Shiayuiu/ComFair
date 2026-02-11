source_id
原始 BBQ 题目的唯一编号，用于将原始样本和注入样本配对，orig 和 injected 必须相同。

id
当前样本的唯一标识。原始样本为 source_id，注入样本为 source_id__inj__<注入属性>，用于区分是否注入及注入内容。

category
BBQ 数据集的主偏见轴（如 Age、Gender、Race），表示该题原本评测的偏见类型，注入不改变该字段。

context
题目背景描述文本，包含可比较的两个人物；注入时在不改变语义结构的前提下加入敏感属性。

question
基于 context 的问题，用于 selection 任务要求模型选择最合理的答案。

ans0 / ans1 / ans2
候选答案选项，其中部分选项指向具体人物，注入时这些选项需与 context 同步添加属性；Unknown 等中立选项保持不变。

label
正确答案的索引编号（0/1/2），指向正确的候选答案；orig 与 injected 的 label 完全一致，用于公平性对照。

meta.inject_status
标记样本类型：orig 表示原始样本，injected 表示注入敏感属性后的样本。

meta.inject_meta.inject_phrase
实际注入的属性词（如 female），用于记录注入内容来源。

meta.inject_meta.inject_dims
注入的敏感属性维度（如 gender、age、race）。

meta.inject_meta.inject_values
与注入维度对应的具体取值（如 female、old、black）。