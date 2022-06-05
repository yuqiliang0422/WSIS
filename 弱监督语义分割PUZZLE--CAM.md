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
## 分析：
这个方法为什么要这么做，能work的原因是什么？有什么优点，有什么缺点，值得我们后续工作借鉴吗，如果值得，能复现文中的结果吗？
## 1. 这个方法为什么想到这么去做？文中Introduction里有一句话：The main limitation of WSSS is that the process of generating pseudo-labels from CAMs that use an image classifier is mainly focused on the most discriminative parts of the objects.即整张图片由分类任务产生的CAMs图主要关注物体最具辨别力的部分。然后作者发现，用图像块产生的CAM图拼接后的大CAM图关注的物体区域更大。如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/a65332a5fe604f45b62baf7fd7fb6958.png#pic_center)

至于为什么会更大，原文中有这样一段话：When generating the CAM of the same class for image patches, the model focuses on finding the key features of the class using only the part of the object. Thus, the merged CAM of image patches highlights the object area more accurately than the CAM of a single image.

## 2. 能work的原因是什么?
首先，由image patch产生的merged CAM比image产生的CAM好；其次，缩小merged CAM和整张图片的CAM的差距，缩小特征之间的差距，使得训练好的模型直接产生的整幅图片的CAM更好。

## 3. 有什么优点：(1)从patch入手，CNN网络对每个patch的关注区域更大了，汇总到一起对整个物体的关注区域更大了。(2)损失函数中加入merged CAM和CAM的差距一项，可以让网络学习去产生更完整的整幅图像的CAM。消融表格如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/502879ad7feb4071bcc883798c14faa7.png#pic_center)

## 4. 有什么缺点：(1)文中只将原图分成了4个patch,没有划分不同规格patch。
(2)如果merged CAM和CAM对物体的某个区域的激活都很小，对损失函数没有影响。

## 5. 借鉴之处：
(1) 尝试在Vision transformer中以图像块为单位生成激活图。
(2) 根据文中所述，单个图像块产生的CAM也能聚集到物体的一部分上，且可能比整个图像产生的CAM范围更大。那么在WS-RCNN中，分数高的proposal所覆盖的所有图像块集合产生的CAM应该也能聚集到一个实例的一部分上。
(3) 把图像中对应候选区域的部分留下，图像其他地方变为平均灰度值，送入ViT网络，产生CAM图后得到proposal分数，是否也可以进行分类。











