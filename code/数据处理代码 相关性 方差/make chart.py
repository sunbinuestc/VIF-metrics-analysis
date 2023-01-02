import pandas as pd
#from pandas_datareader import data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

image_pairs = range(1 , 22)
image_pairs = [str(x) for x in list(image_pairs)]
x = range(len(image_pairs))
## q ab/f data
CBF = [0.5568,0.4345,0.6172,0.7443,0.519,0.814,0.6,0.7305,0.4516,0.494,0.7521,0.5081,0.724,0.5827,0.5488,0.4925,
       0.561,0.7196,0.4869,0.4346,0.3687]
CNN = [0.6685,0.5855,0.7025,0.8048,0.5835,0.8616,0.5626,0.7909,0.5676,0.5839,0.8054,0.6225,0.7906,0.6504,0.5721,
       0.6077,0.6402,0.775,0.498,0.5342,0.6015]
GFF = [0.6698,0.6219,0.6944,0.8096,0.6192,0.8501,0.6125,0.798,0.581,0.576,0.8137,0.3763,0.8057,0.6532,0.5845,0.351,
       0.6592,0.7704,0.6047,0.3265,0.3311]
HMSD_GF = [0.6532,0.5637,0.6476,0.7777,0.5338,0.7572,0.6066,0.7644,0.5532,0.5448,0.7716,0.5868,0.765,0.6276,0.5513,
           0.556,0.5881,0.7319,0.4807,0.4913,0.5348]
Hybrid_MSD = [0.6663,0.5719,0.6578,0.7996,0.5306,0.8572,0.6049,0.7754,0.5497,0.5471,0.7937,0.5918,0.7768,0.6327,
              0.5537,0.5585,0.6091,0.7596,0.4931,0.492,0.5338]
MGFF = [0.5856,0.5152,0.5815,0.7236,0.4886,0.7709,0.5871,0.6834,0.485,0.508,0.6978,0.5126,0.6619,0.5819,0.5102,
        0.5016,0.5517,0.6671,0.4348,0.4579,0.5176]
MST_SR = [0.6794,0.6005,0.7115,0.8043,0.5921,0.8653,0.602,0.7959,0.5713,0.5816,0.8027,0.6269,0.7933,0.6537,
          0.5482,0.6116,0.6551,0.772,0.4872,0.5391,0.589]
NSCT_SR = [0.6881,0.5331,0.6799,0.8186,0.5627,0.848,0.6156,0.7922,0.542,0.552,0.8257,0.5904,0.7963,0.641,
           0.5662,0.5649,0.6306,0.7827,0.5571,0.4968,0.473]
RP_SR = [0.6352,0.5228,0.584,0.7569,0.4243,0.8098,0.4753,0.7504,0.5288,0.4888,0.7639,0.5208,0.7582,0.5915,
         0.4619,0.4988,0.4855,0.7063,0.4022,0.3961,0.3197]
TIF = [0.5815,0.522,0.588,0.7518,0.5076,0.8031,0.5726,0.7199,0.4993,0.5163,0.7432,0.5088,0.7065,0.5868,
       0.5052,0.5171,0.5466,0.6934,0.4176,0.4584,0.5221]

# ADF = [60.066,58.248,56.884,61.296,58.65,58.3,58.218,61.753,61.15,59.425,59.647,55.325,60.904,59.433,
# 55.924,55.643,56.605,59.4,55.888,58.064,55.69]
# DLF = [60.07,58.253,56.892,61.319,58.678,58.322,58.226,61.805,61.176,59.767,59.661,55.337,60.938,59.455,
# 55.93,55.661,56.637,59.414,55.914,58.13,55.734]
# FPDE = [60.068,58.25,56.889,61.307,58.672,58.314,58.224,61.777,61.156,59.707,59.659,55.331,60.92,59.444,
# 55.927,55.651,56.63,59.41,55.9,57.489,55.712]
# IFEVIP = [58.825,57.492,55.499,59.928,57.306,57.504,57.316,60.237,59.722,57.796,58.545,54.176,59.818,58.112,
# 54.435,54.498,55.237,58.197,54.525,57.012,54.467]
# LatLRR = [57.724,56.346,55.215,58.371,56.168,56.788,56.53,58.631,58.198,56.218,57.492,53.462,58.351,57.109,
# 53.898,53.845,54.778,57.153,53.827,55.543,54.124]
# MGFF = [59.999,57.986,56.689,61.093,58.212,58.169,58.009,61.456,61.006,59.434,59.519,55.195,60.723,59.33,55.396,
#         55.476,56.463,59.277,55.625,57.742,55.656]
# MSVD = [60.061,58.247,56.874,61.283,58.633,58.301,58.203,61.742,61.169,59.747,59.622,55.323,60.901,59.436,55.85,
# 55.642,56.618,59.386,55.856,58.089,55.731]
# ResNET = [60.059,58.251,56.891,61.309,58.678,58.317,58.216,61.808,61.173,59.768,59.658,55.335,60.926,59.454,55.932,
# 55.66,56.636,59.414,55.913,58.132,55.732]
# TIF = [59.967,58.131,56.698,61.081,58.371,58.145,58.02,61.434,61.015,59.482,59.475,55.251,60.657,59.32,55.553,
# 55.461,56.361,59.259,55.64,57.798,55.607]
# VSMWLS = [59.671,58.187,56.712,60.978,58.326,58.213,57.525,61.569,60.992,59.601,59.308,55.169,60.71,59.053,
# 55.645,55.455,56.401,59.218,55.72,57.926,55.686]
#plt.plot(x, CBF, marker='o', mec='r', mfc='w', label='CBF')#mec 颜色 mfc=w 空洞
#plt.plot(x, CNN, marker='*', ms=10, label='CNN')   #ms 改变大小
plt.plot(x, CBF, marker='o', mfc='w', color = 'red', ms=4, label='CBF:0.579')
plt.plot(x, CNN, marker='*', mfc='w', color = 'orange',ms=4, label='CNN:0.658')
plt.plot(x, GFF, marker='s', mfc='w', color = 'blue', ms=4, label='GFF:0.624')
plt.plot(x, HMSD_GF, marker='v', mfc='w', color = 'purple',ms=4, label='HMSD_GF:0.623')
plt.plot(x, Hybrid_MSD, marker='<', mfc='w',color = 'magenta', ms=4, label='Hybrid_MSD:0.636')
plt.plot(x, MGFF, marker='*', mfc='w',color = 'limegreen', ms=4, label='MGFF:0.573')
plt.plot(x, MST_SR, marker='+', mfc='w', color = 'maroon',ms=4, label='MST_SR:0.661')
plt.plot(x, NSCT_SR, marker='.', mfc='w', color = 'darkturquoise',ms=4, label='NSCT_SR:0.646')
plt.plot(x, RP_SR, marker='o', mfc='w', color = 'chocolate',ms=4, label='RP_SR:0.566')
plt.plot(x, TIF, marker='s', mfc='w', color = 'thistle',ms=4,label='TIF:0.584')

# plt.plot(x, ADF, marker='o', mfc='w', color = 'red', ms=4, label='ADF:58.405')
# plt.plot(x, DLF, marker='*', mfc='w', color = 'orange',ms=4, label='DLF:58.444')
# plt.plot(x, FPDE, marker='s', mfc='w', color = 'blue', ms=4, label='FPDE:58.402')
# plt.plot(x, IFEVIP, marker='v', mfc='w', color = 'purple',ms=4, label='IFEVIP:57.174')
# plt.plot(x, LatLRR, marker='<', mfc='w',color = 'magenta', ms=4, label='LatLRR:56.180')
# plt.plot(x, MGFF, marker='*', mfc='w',color = 'limegreen', ms=4, label='MGFF:58.212')
# plt.plot(x, MSVD, marker='+', mfc='w', color = 'maroon',ms=4, label='MSVD:58.415')
# plt.plot(x, ResNET, marker='.', mfc='w', color = 'darkturquoise',ms=4, label='ResNET:58.441')
# plt.plot(x, TIF, marker='o', mfc='w', color = 'chocolate',ms=4, label='TIF:58.225')
# plt.plot(x, VSMWLS, marker='s', mfc='w', color = 'thistle',ms=4,label='VSMWLS:58.194')

plt.legend(bbox_to_anchor = (1.05,1),loc = 'upper left', borderaxespad=0 )  # 让图例生效
plt.xticks(x, image_pairs, rotation=1)
plt.rcParams['font.sans-serif']=['SimSun'] #用来正常显示中文标签
# config = {
#     "font.family":'serif',
#     "font.size": 18,
#     "mathtext.fontset":'stix',
#     "font.serif": ['SimSun'],
# }
# rcParams.update(config)

plt.margins(0)
plt.subplots_adjust(bottom=0.10)
plt.xlabel('图像对',fontsize=14)  # X轴标签
plt.ylabel("指标值",fontsize=14)  # Y轴标签
plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# plt.yticks([ 52 ,54, 56, 58, 60, 62, 64])
#plt.title("$Q^{AB/F}$", fontsize=20) #标题
# plt.title("PSNR", fontsize=20) #标题
x_major_locator= MultipleLocator(2) #设置间隔刻度
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.subplots_adjust(right=0.7)
plt.savefig('D:\\Qabf.png', dpi=900)
plt.show()
