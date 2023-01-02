# <center>可见光与红外图像融合质量评价指标分析</center>

针对可见光与红外图像融合，基于VIFB提出了一种基于统计分析的客观评价指标分析方法，该方法可以推广至更多的图像融合应用，指导选择具有代表性的客观评价指标。当前使用了VIFB集成的21组图像、20种融合算法以及13种评价指标进行测试。并对结果采用 Kendall 相关系数分析融合指标间的相关性，聚类得到指标分组；采用 Borda 计数排序法统计算法的综合排序，分析单一指标排序和综合排序的相关性，得到一致性较高的指标集合；采用离散系数分析指标均值随不同算法的波动程度，选择充分体现不同算法间差异的指标；综合相关性分析、一致性分析及离散系数分析，总结具有代表性的建议指标集合。



## 评价指标集合

在 VIFB 中集成了 13 个评估指标。这些代码是从互联网、论坛等收集的，并检查了作者。

1. Avgerage gradient
2. Cross entropy
3. Edge intensity
4. Entropy
5. Mutual information
6. PSNR
7. Qabf
8. Qcb
9. Qcv
10. RMSE
11. Spatial frequency
12. SSIM
13. SD 

### 如何计算评价指标

1. 请在 VIFB-master\util\configMetrics.m 中设置您要计算的指标。
2. compute_metrics.m 用于计算评估指标。请在compute_metrics.m 中更改输出路径。

## Kendall相关系数

打开数据处理代码 相关性 方差中的main.m，需要限定输入源数据路径以及在表中的范围。并且更改输出路径。

## 论文引用格式

Sun B, Gao Y X, Zhuge W W and Wang Z X. 2023. Analysis of quality objective assessment metrics for visible and infrared image fusion. Journal of Image and Graphics,28(01):0000-0000(孙彬,高云翔,诸葛吴为,王梓萱. 2023. 可见光与红外图像融合质量评价指标分析. 中 国图象图形学报,28(01):0000-0000)[DOI:10. 11834 / jig. 210719]







