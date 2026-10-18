# Corporate Financial Distress Classification 中文研究导读

本项目利用美国公司的财务报表，对数据集中标记为 failed 的公司年度记录进行分类和风险排序。它是原 UCLA MFE 413 团队项目之上的个人研究扩展，由 Qidian (Martin) Xing 主导，并披露 Codex 在实现、实验和文档方面的辅助。

[中文研究 PDF](Research_Guide_CN.pdf) 为原八页报告，逐字节保留。PDF 内的历史研究阶段标签用于追溯来源，不代表需要读者选择的多个正式版本。

## 如何理解结果

冻结的 XGBoost 与 MLP 年度排名等权融合，在已被早期研究接触过的 2015–2018 年历史记录上，mean annual AP 为 **0.373320**，pooled ROC-AUC 为 **0.921602**，mean annual Recall@Top 5% 为 **61.98%**。同训练截止期最强固定基线的年度 AP 为 **0.302835**，差值为 **+0.070485**。不能用旧研究不同口径的 pooled AP 0.206 作为直接改进分母。

## 为什么不改用历史成绩更高的 MLP

开发期的五年、三种子比较中，等权融合年度 AP 为 0.581580，MLP 单独排名为 0.538859，因此融合在一次 2014 年确认与历史评分前已冻结。历史期 MLP 单独年度 AP 为 0.392916，高于融合 0.373320；保留这一反转，正式结果继续采用事先选定的融合模型。否则就会变成看完历史成绩再选模型。[选择记录](../docs/MODEL_SELECTION.md)

## 研究真正支持什么

时间验证、训练期内预处理、多种子训练和同截止期基线使比较更清晰。三年半衰期权重在相同开发年份和种子下帮助两个最终模型。复杂特征、数值嵌入、扩大网络和 ranking loss 并非普遍有效。重采样增加 epoch 内更新次数，因此必须同时检查实际 optimizer updates 和类别曝光量。失败、未完成和负面实验仍保存在公开聚合表中。

数据共有 78,682 条 firm-year、8,971 家公司，但公司内部 failed/alive 标签始终固定，缺乏真实破产日期和报表发布日期。历史期并非未接触的新测试集。最终截止期自然未见公司的 failed 样本只有五家。因此这些结果不能被描述为已验证的一年期破产预测、年度 PD 或稳定的前瞻泛化。

继续阅读：[完整结果表](../results/MODEL_COMPARISON.md)、[已执行 Notebook](../notebooks/02_Model_Comparison.ipynb)、[复现说明](../docs/REPRODUCTION.md)、[贡献与辅助声明](../docs/CONTRIBUTIONS.md)。
