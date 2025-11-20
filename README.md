# Intersectional_Fairness_alpha

本代码仓库用于存储以Intersectional_Fairness为主题的调研结果，code和工作流（暂定）

1\. 目前工作：调研

关于模型文本生成的公平性指标，关于图像生成的公平性指标；

对于常见任务（生成任务、分类任务等）的评测标准；

如何建立一个有区分度的Intersectional-fairness的指标

2.下一步？code demo

代码方案：采用CEB、Opencompass的repo作为参考。

Opencompass的框架：数据集、模型（开源、闭源）接口；对于生成类任务，分别保留infer、eval阶段的数据（将infer阶段的数据保留，可以通过自行接入的evaluator得到评测结果；对于默认的结果，生成指标的表格。
