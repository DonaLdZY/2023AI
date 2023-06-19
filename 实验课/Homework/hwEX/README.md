# 决策树

## 数据集

给定一个幼儿园儿童信息数据集，该数据集包含 12960 个入学儿童的自身及家庭状况，每个样本包含 8 个特征，目标是找到决策树模型可以将这些数据拟合，从而对是否适合入学做出预测分析，输出类别共 5 个。

特征：'parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class'

分类：'not_recom', 'priority', 'recommend', 'spec_prior', 'very_recom'

[Nursery - UCI Machine Learning Repository](http://archive.ics.uci.edu/dataset/76/nursery)

数据集下载后可以得到 `.data` 和 `.names` 文件，部分数据处理提示已在 `.ipynb` 文件中给出，可作为参考。

## 作业内容

### 基础部分

- 给出对于数据集特征的分析，例如样本分布等；
- 实现基于信息增益 (IG) 和信息增益比例 (IGR) 的决策树构造算法，比较性能区别并分析；
- 对基于 IG 和 IGR 构造的决策树实现预剪枝策略，比较二者对于最终测试准确率的影响。

### 附加部分

对基于 IG 和 IGR 构造的决策树实现后剪枝策略，比较二者对于最终测试准确率的影响。

本次作业不要求提交。
