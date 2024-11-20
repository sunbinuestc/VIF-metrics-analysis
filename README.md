# 可见光与红外图像融合质量评价指标分析
<a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/README_eng_ver.md">English Version</a>
## 目的
客观评价作为图像融合的重要研究领域，是评价融合算法性能的有力工具。目前，已有几十种不同类型的评价指标，但各应用领域包括可见光与红外图像融合，仍缺少统一的选择依据。为了方便比较不同融合算法性能，提出一种客观评价指标的通用分析方法并应用于可见光与红外图像融合。
## 方法
将可见光与红外图像基准数据集中的客观评价指标分为两类，分别是基于融合图像的评价指标与基于源图像和融合图像的评价指标。采用Kendall相关系数分析融合指标间的相关性，聚类得到指标分组；采用Borda计数排序法统计算法的综合排序，分析单一指标排序和综合排序的相关性，得到一致性较高的指标集合；采用离散系数分析指标均值随不同算法的波动程度，选择充分体现不同算法间差异的指标；综合相关性分析、一致性分析及离散系数分析，总结具有代表性的建议指标集合。

  <table width="723.73" border="0" cellpadding="0" cellspacing="0" style='width:542.80pt;border-collapse:collapse;table-layout:fixed;'>
   <col width="90.87" style='mso-width-source:userset;mso-width-alt:2907;'/>
   <col width="221.47" style='mso-width-source:userset;mso-width-alt:7086;'/>
   <col width="37.67" style='mso-width-source:userset;mso-width-alt:1205;'/>
   <col width="100" style='mso-width-source:userset;mso-width-alt:3200;'/>
   <col width="136.87" span="2" style='mso-width-source:userset;mso-width-alt:4379;'/>
   <tr height="18" style='height:13.50pt;'>
    <td class="xl65" height="18" width="350" colspan="3" style='height:13.50pt;width:262.50pt;border-right:.5pt solid windowtext;border-bottom:.5pt solid windowtext;' x:str>Metrics</td>
    <td class="xl67" width="100" style='width:75.00pt;' x:str>Key formula</td>
    <td class="xl66" width="136.87" style='width:102.65pt;' x:str>Parameter Values</td>
    <td class="xl66" width="136.87" style='width:102.65pt;' x:str>代码链接</td>
   </tr>
   <tr height="36" style='height:27.00pt;'>
    <td class="xl68" height="126" rowspan="5" style='height:94.50pt;border-right:.5pt solid windowtext;border-bottom:.5pt solid windowtext;' x:str>基于融合图像质量的评价指标</td>
    <td class="xl69" x:str>标准差（standard deviation）</td>
    <td class="xl66" x:str>SD</td>
    <td class="xl66" x:str>$$SD=\sqrt {\sum ^{M}_{i=1} {\sum ^{N}_{j=1} {{(F(i,j)-\mu )}^{2}}}}$$</td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsVariance.m" target="_parent">SD</a></td>
   </tr>
   <tr height="18" style='height:13.50pt;'>
    <td class="xl69" x:str>边缘强度（edge intensity）</td>
    <td class="xl66" x:str>EI</td>
    <td class="xl66" x:str>$$EI=\frac {\sqrt {\sum ^{M}_{i=1} {\sum ^{N}_{j=1} {({s}_{x}{(i,j)}^{2}+{s}_{y}{(i,j)}^{2})}}}} {M\times N}$$</td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsEdge_intensity.m" target="_parent">EI</a></td>
   </tr>
   <tr height="18" style='height:13.50pt;'>
    <td class="xl69" x:str>熵（entropy）</td>
    <td class="xl66" x:str>EN</td>
    <td class="xl66" x:str>$$E N_{F}=-\sum_{f=1}^{n} p_{f} \log p_{f}$$</td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsEntropy.m" target="_parent">EN</a></td>
   </tr>
   <tr height="36" style='height:27.00pt;'>
    <td class="xl69" x:str>平均梯度（average gradient）</td>
    <td class="xl66" x:str>AG</td>
    <td class="xl66" x:str>$$A G=\frac{1}{(M-1)(N-1)} \times \sum_{i=1}^{M-1} \sum_{j=1}^{N-1} \times 
\sqrt{\frac{(F(i+1 ,j)-F(i ,j))^{2}+(F(i, j+1)-F(i ,j))^{2}}{2}}$$</td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsAvg_gradient.m" target="_parent">AG</a></td>
   </tr>
   <tr height="18" style='height:13.50pt;'>
    <td class="xl69" x:str>空间频率（space frequency）</td>
    <td class="xl66" x:str>SF</td>
    <td class="xl66" x:str>$$S F=\sqrt{\left(F_{R}\right)^{2}+\left(F_{C}\right)^{2}+\left(F_{M D}\right)^{2}+\left(F_{S D}\right)^{2}}$$</td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsSpatial_frequency.m" target="_parent">SF</a></td>
   </tr>
   <tr height="36" style='height:27.00pt;'>
    <td class="xl68" height="224" rowspan="8" style='height:168.00pt;border-right:.5pt solid windowtext;border-bottom:.5pt solid windowtext;' x:str>基于源图像和融合图像的评价指标</td>
    <td class="xl69" x:str>均方根误差 （root mean squared error）</td>
    <td class="xl66" x:str>RMSE</td>
    <td class="xl66" x:str>$$R M S E=\frac{R M S E_{A F}+R M S E_{B F}}{2}$$</td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsRmse.m" target="_parent">RMSE</a></td>
   </tr>
   <tr height="36" style='height:27.00pt;'>
    <td class="xl69" x:str>峰值信噪比（peak signal to noise ratio)</td>
    <td class="xl66" x:str>PSNR</td>
    <td class="xl66" x:str>$$\text { PSNR }=10 \lg \frac{[\max (F(i, j))-\min (F(i, j))]^{2}}{M S E}$$</td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsPsnr.m" target="_parent">PSNR</a></td>
   </tr>
   <tr height="18" style='height:13.50pt;'>
    <td class="xl69" x:str>交叉熵（cross entropy）</td>
    <td class="xl66" x:str>CE</td>
    <td class="xl66" x:str>$$C E=\sqrt{\frac{\left(C E_{F}^{A}\right)^{2}+\left(C E_{F}^{B}\right)^{2}}{2}}$$</td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsCross_entropy.m" target="_parent">CE</a></td>
   </tr>
   <tr height="36" style='height:27.00pt;'>
    <td class="xl69" x:str>互信息（mutual information）</td>
    <td class="xl71" x:str>MI</td>
    <td class="xl66" x:str>$$M I_{F}^{A B}=I_{F A}(f ; a)+I_{F B}(f ; b)$$</td></td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsMutinf.m" target="_parent">MI</a></td>
   </tr>
   <tr height="20" style='height:15.00pt;'>
    <td class="xl69" x:str>边缘保持度</td>
    <td class="xl71" x:str>Q<font class="font22"><sup>AB/F</sup></font></td>
    <td class="xl66" x:str>$$Q^{A B / F}=\frac{\sum_{i=1}^{N} \sum_{j=1}^{M} Q^{A F}(i ,j) \omega^{A}(i ,j)+Q^{B F}(i ,j) \omega^{B}(i ,j)}{\sum_{i=1}^{N} \sum_{j=1}^{M}\left(\omega^{A}(i ,j)+\omega^{B}(i ,j)\right)}$$</td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsQabf.m" target="_parent">Q<font class="font21"><sup>AB/F</sup></font></a></td>
   </tr>
   <tr height="36" style='height:27.00pt;'>
    <td class="xl69" x:str>结构相似度的均方根误差（root mean squared error）</td>
    <td class="xl66" x:str>SSIM</td>
    <td class="xl66" x:str>$$S S I M_{A F}=l(\boldsymbol{A}, \boldsymbol{F}) \times c(\boldsymbol{A}, \boldsymbol{F}) \times s(\boldsymbol{A}, \boldsymbol{F})$$</td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsSsim.m" target="_parent">SSIM</a></td>
   </tr>
   <tr height="21" style='height:15.75pt;'>
    <td class="xl69" x:str>Chen-Varshney</td>
    <td class="xl71" x:str>Q<font class="font24"><sub>CV</sub></font></td>
    <td class="xl66" x:str>$$Q_{\mathrm{cv}}= 
\frac{\sum_{l=1}^{L}\left(\lambda\left(\boldsymbol{A}^{w_{l}}\right) D\left(\boldsymbol{A}^{w_{l}} ,\boldsymbol{F}^{w_{l}}\right)+\lambda\left(\boldsymbol{B}^{w_{l}}\right) D\left(\boldsymbol{B}^{w_{l}} ,\boldsymbol{F}^{w_{l}}\right)\right)}{\sum_{l=1}^{L}\left(\lambda\left(\boldsymbol{A}^{w_{l}}\right)+\lambda\left(\boldsymbol{B}^{w_{l}}\right)\right)}$$</td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsQabf.m" target="_parent">Q<font class="font23"><sub>CV</sub></font></a></td>
   </tr>
   <tr height="21" style='height:15.75pt;'>
    <td class="xl69" x:str>Chen-Blum</td>
    <td class="xl71" x:str>Q<font class="font24"><sub>CB</sub></font></td>
    <td class="xl66" x:str>$$Q_{\mathrm{GQM}}(i ,j)=\lambda_{A}(i ,j) Q_{A F}(i ,j)+\lambda_{B}(i ,j) Q_{B F}(i ,j)$$</td>
    <td class="xl66"></td>
    <td class="xl70" x:str><a href="https://github.com/sunbinuestc/Analysis-of-quality-objective-assessment-metrics-for-visible-and-infrared-image-fusion/blob/main/code/Matlab/VIFB/metrics/metricsQcb.m" target="_parent">Q<font class="font23"><sub>CB</sub></font></a></td>
   </tr>
   <![if supportMisalignedColumns]>
    <tr width="0" style='display:none;'>
     <td width="91" style='width:68;'></td>
     <td width="221" style='width:166;'></td>
     <td width="38" style='width:28;'></td>
     <td width="100" style='width:75;'></td>
     <td width="137" style='width:103;'></td>
   <![endif]>
    </tr>
<table>
  
## 结果
在13对彩色可见光与红外和8对灰度可见光与红外两组图像源中，分别统计分析不同图像融合算法的客观评价数据，得到可见光与红外图像融合的建议指标集(标准差、边缘保持度)，作为融合算法性能评估的重要参考。相较于现有方法，实验覆盖20种融合算法和13种客观评价指标，并且不依赖主观评价结果。
## 结论
针对可见光与红外图像融合，提出了一种基于统计分析的客观评价指标分析方法，该方法可以推广至更多的图像融合应用，指导选择具有代表性的客观评价指标。

<div align=center><img src="assets/1.png"></div>

<div align=center>图1</div>

图像融合客观指标分析的流程如图1所示。通过相关性分析和聚类得到指标分组，设计不受分组干扰的投票法统计算法的综合排序，分析单一指标排序和综合排序的相关性，得到一致性较高的指标集合；利用离散程度分析指标随不同算法的波动程度，选择充分体现不同算法间差异的指标。在图像融合实验基础上，综合相关性分析、一致性分析及离散程度分析，得到适用于可见光和红外图像融合的{建议指标集合}。

针对有限样本在实验设定下得到的建议指标集合是非排他性的指标建议,即选择多个指标从不同角度综合评价融合结果时，建议选择而非只选择的指标集合。相关领域研究者可将方法推广至多聚焦图像、医学图像以及遥感图像融合，得到适用于不同应用场景的图像融合客观评价指标建议。
## 代码使用方法
### Matlab
在`code/Matlab`文件夹下，指标计算代码存放在`VIFB`文件夹中，并配有相关说明；指标分析代码存放在`metric analysis`文件夹中，需修改`xlsread`、`xlswrite`函数处理的文件路径。在读取的xlsx文件中应按照下列格式存放各方法的各种指标。

<table>
  <tr>
    <th></th>
    <th>EN</th>
    <th>SSIM</th>
    <th>PSNR</th>
    <th>......</th>
    <th>Borda rank</th>
  </tr>
  <tr>
    <th>DenseFuse</th>
    <td>xxx</td>
    <td>xxx</td>
    <td>xxx</td>
    <td>xxx</td>
    <td>xxx</td>
  </tr>
  <tr>
    <th>DIDFuse</th>
    <td>xxx</td>
    <td>xxx</td>
    <td>xxx</td>
    <td>xxx</td>
    <td>xxx</td>
  </tr>
  <tr>
    <th>......</th>
    <td>xxx</td>
    <td>xxx</td>
    <td>xxx</td>
    <td>xxx</td>
    <td>xxx</td>
  </tr>
</table>

### Python
在`code/Python`文件夹下，指标计算代码存放在`metric calculation`文件夹中，指标分析代码存放在`metric analysis`文件夹中，均配有相关说明。

## 引用信息
孙彬, 高云翔, 诸葛吴为, 王梓萱. 可见光与红外图像融合质量评价指标分析[J]. 中国图象图形学报, 2023,28(1):144-155. DOI： 10.11834/jig.210719.

Bin Sun, Yunxiang Gao, Wuwei Zhuge, Zixuan Wang. Analysis of quality objective assessment metrics for visible and infrared image fusion[J]. Journal of Image and Graphics, 2023,28(1):144-155. DOI： 10.11834/jig.210719.

中国图象图形学报官方链接<a href="https://www.cjig.cn/zh/article/doi/10.11834/jig.210719/">可见光与红外图像融合质量评价指标分析 (cjig.cn)</a>

