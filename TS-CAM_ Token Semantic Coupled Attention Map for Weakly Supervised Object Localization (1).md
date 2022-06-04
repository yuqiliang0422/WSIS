# TS-CAM：由Vision Transformer架构产生CAM类别激活图的一种方法
@[TOC](文章目录)

# 前言

《TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised
Object Localization》这篇文章解决的是弱监督定位问题(单标签)，但其主要内容是由Vision Transformer架构得到一种类别激活图: TS-CAM。本文结合论文内容和Vision transformer, TS-CAM的代码，用画图的形式解释作者的思路。 

---


# 一、论文链接和代码链接以及重新训练模型得到可视化结果

TS-CAM论文：[
论文](https://arxiv.org/pdf/2103.14862v5.pdf)
TS-CAM代码链接：[
TS-CAM代码](https://github.com/vasgaowei/TS-CAM)
Vision Transformer代码链接:[
ViT代码](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer)
以下内容均来自于对以上文献和代码的理解，在此表示感谢！
在服务器上重新训练模型后，我们重新可视化CUB200种鸟类数据集的结果，确保代码应该无误。
以下是原文中的一张鸟类结果图。最后一张图是该图片对应的TS-CAM。
![在这里插入图片描述](https://img-blog.csdnimg.cn/5703317d69294909a7e6481ef58ddb96.jpeg#pic_center)
下面是重新训练模型后得到的该张图片的TS-CAM结果图(取阈值0.4)。
![在这里插入图片描述](https://img-blog.csdnimg.cn/c9d3f53c005c4b3aa71bdcc76e7946bf.jpeg#pic_center)
以下是另一张原文中的结果图。最后一张图是该图片对应的TS-CAM。
![在这里插入图片描述](https://img-blog.csdnimg.cn/b8ec6981f4da47ec81bb95ee04eee1a0.jpeg#pic_center)
下面是重新训练模型后得到的该张图片的TS-CAM结果图(取阈值0.4)。
![在这里插入图片描述](https://img-blog.csdnimg.cn/7539595704d34c3b80bc2216fc3a6b53.jpeg#pic_center)
确认代码运行无误后，我们结合代码看作者的思路。

# 二、TS-CAM论文是如何由Vision Transformer得到类别激活图的？
## 1.embedding层
我们以一种规模的ViT模型vit_base_patch16_224_in21k为例，16表示每个patch大小是16×16个像素，224表示所有图像要先调整尺寸为224×224后再输入模型，in21k表示该模型已在Imagenet21k上预训练过。对于一个大小为[3, 224,224]的图像，每个patch大小为16×16，则一幅图像中有14*14个patch。
对于transformer模块，要求输入的是token序列(向量序列)，得到每一个图像块的token是用卷积操作实现的。例如，要将每个patch编码为一个768长度的向量，则我们需要768个卷积核，每个卷积核大小为16×16，步长是16。如下图所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/f5a8c47ac8cd4b03bc6398735b482e38.jpeg#pic_center)
每一行是一个patch的token向量。
在输入到transformer block之前需要加上class token以及position embedding,这两个都是可训练参数。class token是作为一个单独的768维向量加到第一行上，position embedding是作为一个768维向量加进每一个token中，如此，embedding层最终得到的矩阵大小是[197,768], 第一行对应class token，其余行对应每一个patch token。
## 2.transformer encoder层+Conv2d head层
在得到class token和patch token后，我们就得到了embedded patches。将其送入transformer encoder中，transformer encoder的内部结构如下图所示: 
![在这里插入图片描述](https://img-blog.csdnimg.cn/242b1c5d27df4079b56221b6448ddb3d.jpeg#pic_center)
除了Layer Norm和Dropout外，就是Multi-Head Attention和MLP两个模块了。论文中有用到12个transformer block，每个Multi-Head Attention模块里用到12个head。我们只看一个transformer block, 并以3个head为例，multi-head attenttion操作如下:
![在这里插入图片描述](https://img-blog.csdnimg.cn/540b1ee457da47819b2b24962ecc1e61.jpeg#pic_center)
论文中有12个head, 所以对每个head而言，patch token(class token)的特征向量维度是768÷12=64维。对应的$q$矩阵和$k^{\top}$矩阵如下图所示:
![在这里插入图片描述](https://img-blog.csdnimg.cn/bfd1638bc64446c5b990c7a3d07c9a3e.jpeg#pic_center)

根据公式$q \cdot k^{\top} / \sqrt{d k}$计算注意力矩阵attention matrix,然后以行为单位进行softmax操作。
![在这里插入图片描述](https://img-blog.csdnimg.cn/a7fdee7f788e41489451808fc5a3c852.jpeg#pic_center)
矩阵中每一个元素是一个patch对另一个patch的特征依赖($q$和$k$之间的相似度)
然后我们将attention matrix和$v$矩阵相乘，得到每个patch的新的特征向量。操作如下图:
![在这里插入图片描述](https://img-blog.csdnimg.cn/39e14b209e3d48bd8186bb6dbf757ddf.jpeg#pic_center)
从上图可以看到，对于一个head而言，每一列是一个patch(class)的64维特征向量，我们将12个head的特征向量融合，融合的方式就是乘上一个矩阵。操作如下图所示:
![在这里插入图片描述](https://img-blog.csdnimg.cn/737272a3e81646f9af6b406839645bea.jpeg#pic_center)
上图中197×768的矩阵是经过一个多头注意力后得到的class token和patch token。这之后是一个MLP模块，里面是Linear层和GELU激活层，这里略。
之后的transformer block重复上述操作。在最后一个transformer block结束后，我们得到197×768大小的class token+patch token矩阵，如上图所示，我们将图像块的特征向量即patch token部分取出来，记作x_patch,我们将它按照在图像中的位置进行reshape，操作如下图所示:
![在这里插入图片描述](https://img-blog.csdnimg.cn/0921a340fa3144d5a2e336b187d660a6.jpeg#pic_center)
然后经过一个卷积层，将通道数变为类别数，再由全局平均池化得到最终的类别分数。卷积层之后的特征层论文中记为feature_maps。作者在文中提到，这个feature_maps富含语义信息，可以和不含语义信息的注意力图相乘得到类别激活图。注意力图由我们在每个transformer block(一共12个)得到的attention matrix获得。具体做法如下图;
![在这里插入图片描述](https://img-blog.csdnimg.cn/ac53824484084d459df90a6a7ac2e609.jpeg#pic_center)
在得到注意力矩阵joint_attention后，作者只取了第一行(第一行第一列的元素也不要)向量，叫做class token的注意力向量，将这个向量按照patch在图像中的位置进行reshape操作，得到cams_re, 如下图所示:
![在这里插入图片描述](https://img-blog.csdnimg.cn/64b291300f784f28b217084f79cd6855.jpeg#pic_center)
最终的TS-CAM图由缺少语义信息的注意力图cams_re和富含语义信息的feature maps按元素相乘得到。
## 3.可视化
下面我们可视化几张COCO数据集上图片的注意力图cams_re和特征图feature maps，以及将它们两个相乘得到的TS-CAM图，COCO数据集有80个类别，所以feature maps和TS-CAM图都是80个通道，而注意力图是单通道的。
示例1:![在这里插入图片描述](https://img-blog.csdnimg.cn/8c999c8d498143d49308b9c27c6658b9.png#pic_center)
将注意力图和特征图相乘即得到下图的TS-CAM类别激活图:
![在这里插入图片描述](https://img-blog.csdnimg.cn/2df040697a93436dbf6b3bb893c186f0.png#pic_center)
红色框是预测出的类别
示例2:
![在这里插入图片描述](https://img-blog.csdnimg.cn/d835f60175664ebeb45051b735a6c68f.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/d5e03748268e4f80ae2a66cf5b288911.png#pic_center)

---

# 总结
以上是TS-CAM的具体获取细节。更多的可视化方法可参考github官方文件:
[CUB数据集可视化TS-CAM](https://github.com/vasgaowei/TS-CAM/blob/master/tools_cam/visualization_attention_map_cub.ipynb)


