@[TOC](弱监督语义分割PUZZLE-CAM)

# PUZZLE-CAM：通过匹配部分和全部特征改进定位能力
原文链接：[原文](https://arxiv.org/pdf/2101.11253v4.pdf)
代码链接：[代码](https://github.com/OFRIN/PuzzleCAM)
知乎解读链接：[知乎解读](https://zhuanlan.zhihu.com/p/398461767)

## 简要介绍:
将图像分成不重叠的块，求出块的CAM后融合记为A，求出整个图像原始的CAM记为B，设计一个损失函数最小化A和B之间的差异。
## 整体文章架构图:
![在这里插入图片描述](https://img-blog.csdnimg.cn/107dea0ff57145ab83f5fc2ac0ddaab4.jpeg#pic_center)
图片描述的很清晰，求出整个图像的CAM和所有块的CAM，设计一个简单的损失函数来改进最上面分支产生CAM的质量。推理阶段，不使用puzzle模块的分类器，只使用最上面的分支，来生成伪标签训练全监督模型(常规操作)。





