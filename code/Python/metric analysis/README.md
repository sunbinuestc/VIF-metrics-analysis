# 指标分析
## 环境依赖
- numpy
- openpyxl
- pandas
- scipy
## 数据准备
在xlsx文件中按照下列格式存放各方法的各种指标。
<table>
  <tr>
    <th></th>
    <th>EN</th>
    <th>SSIM</th>
    <th>PSNR</th>
    <th>......</th>
  </tr>
  <tr>
    <th>DenseFuse</th>
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
  </tr>
  <tr>
    <th>......</th>
    <td>xxx</td>
    <td>xxx</td>
    <td>xxx</td>
    <td>xxx</td>
  </tr>
</table>

## 路径修改
在`main.py`中修改输入数据文件路径`metric_filepath`、输出文件保存路径`correlation_analysis_filepath`、`consistency_analysis_filepath`、`coefficient_of_variation_analysis_filepath`。
