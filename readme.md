# 激光雷达点云检测

本项目基于Libtorch构建网络，模型训练与推理均采用C++实现，使用ROS可视化

支持使用CUDA并行生成Pillars特征

## 网络结构

Pointpillar点特征提取 + Resnet主干网络 + Centerhead检测头

## 效果

![](pic/demo.png)

## 预训练模型

预训练模型基于KITTI数据集

## 耗时测试

测试环境 i3 12100 + RTX3060 ： `25fps`


| 模块  | 耗时(ms) |
| :----: | :----: |
| 总耗时 | 40 |
| 基于CUDA的Pillar数据预处理 | 0.84|
| Pillar特征提取 | 5.8|
| Resnet主干网络 | 27|
| Centerhead检测头 | 5.7|
| NMS | 0.17|
| 后处理障碍物构建 | 0.04|


## 注意事项

### Libtorch与ROS共存问题
Libtorch需要使用ABI版本，否则编译时会出现找不到ROS相关函数实现的问题