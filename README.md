# 研究内容

## 题目：《基于WIFI-CSI的人体姿态检测》

**Summary：** 对收集WiFi信号的CSI数据利用基于机器学习的方法对人体姿态进行检测\

题目解说：https://zhuanlan.zhihu.com/p/544526139?utm_campaign=&utm_medium=social&utm_oi=872053480418861056&utm_psn=1586119821300813824&utm_source=qq

**Source：** 《RF-Based Fall Monitoring Using Convolutional Neural Networks》

**Publication：** https://aryokee.csail.mit.edu/        http://rfpose.csail.mit.edu/



## CSI Tool

CSI工具安装教程：http://dhalperi.github.io/linux-80211n-csitool/old_installation.html



## CSI Datasets

**人体姿态检测数据集：** https://github.com/zzx030702/Wifi_Human_Gesture_Recognition/tree/master/CSI_Data

**跌倒检测数据集：** https://github.com/zzx030702/Wifi_Human_Gesture_Recognition/tree/master/CSI_Data/External/Fall%20Detection



## Related Paper

​	**Note：以下论文已存放在**https://github.com/zzx030702/Wifi_Human_Gesture_Recognition/tree/master/Paper



#### 1《Deep Learning and Its Applications to WiFi Human Sensing: A Benchmark and A Tutorial》

**Note：** 该论文提供的了数据集以及多种机器学习解决方案进行对比

**Publication：** https://arxiv.org/abs/2207.07859

**Code：** https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark

**Dataset：**(谷歌云盘) https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt

**Related Note：**

​				https://blog.csdn.net/y609532842/article/details/126017303    

​				https://blog.csdn.net/y609532842/article/details/126027614

Widar数据集：是用的BVP数据，第22行是时间，400列是x轴20个数据乘以️y轴20个数据

NTU数据集，用matlab打开，342行是3天线*114子载波，2000列是时间

UT-HAR数据集用Excel打开乱码，需要论文提供的配套代码打开



#### 2《RT-Fall: A Real-Time and Contactless Fall Detection System with Commodity WiFi Devices》

**Note：** 论文出自北京大学张大庆团队，**特点：利用相位差作为输入！**基于WIFI和4G/5G的非接触无线感知，解决方案：提供基于Wifi信号接收机的相位差作为输入数据，并利用v-SVM进行活动分类，模型评估较为良好，但是并未提供代码和数据集

**Publication：** https://ieeexplore.ieee.org/abstract/document/7458198

**Code：** 无

**Dataset**：无

**Related Note**：https://blog.csdn.net/weixin_42232024/article/details/120039123



#### 3《**FallDeFi: Ubiquitous Fall Detection using Commodity Wi-Fi Devices**》

**Note：** 发表于2017年，开源的数据集合代码

**Publication：** https://dl.acm.org/doi/10.1145/3161183

**Code：** https://github.com/zzx030702/Wifi_Human_Gesture_Recognition/tree/master/FallDeFi

**Dataset**：https://mega.nz/#!9tFA1JrR!lEVzKwNRkS3PcOd0ssb8V3tOi0ZA5Gs9EOU0drtFYcg

**Related Note**：



#### 4《Widar3.0: Zero-Effort Cross-Domain Gesture Recognition With WiFi》

**Note：** 提供了数据集(数据集名称：Widar），需要看明白数据集，但是未开源代码

**Publication：** https://dl.acm.org/doi/abs/10.1145/3307334.3326081

**Code：** 下列代码为相关的复现代码，作者并未给出源代码

- 1.https://github.com/zzh606/wifi-csi     作者说明：https://zhuanlan.zhihu.com/p/386379754
- 2.https://github.com/tian-apple/Smallterm

**Dataset：** https://pan.baidu.com/share/init?surl=E-iG3Oo5gYRCXGl8uykuTQ   提取码：4m47

**Related Note：** 略



#### 5《FewSense, Towards a Scalable and Cross-Domain WiFi Sensing System Using Few-Shot Learning》

**Note：** 此文章提供了的小样本学习方法的解决方案可运用到人体跌倒检测上

**Publication：** 略

**Code：** https://github.com/Guolin-Yin/FewSense

**Dataset：** 无



#### 6《**DensePose From WiFi**》

**Note：** 无

**Publication：** 略

**Code：** 无

**Dataset**：无

**Related Note**：https://zhuanlan.zhihu.com/p/599366644?utm_medium=social&utm_oi=872053480418861056&utm_psn=1598280757486833664&utm_source=qq



#### 7《A WiFi-Based Smart Home Fall Detection System Using Recurrent Neural Network》

**Note：** 系统阐述人体跌倒检测系统，此论文提供的方法在《相关论文2》中有提到

**Publication：** 略

**Code：** 无

**Dataset**：无

**Related Note**：略





