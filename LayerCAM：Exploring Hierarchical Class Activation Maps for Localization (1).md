

@[TOC](为定位任务探索分层的类别激活图)

---
文章链接：[http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf](http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf)

代码：[https://github.com/frgfm/torch-cam](https://github.com/frgfm/torch-cam)

[https://github.com/PengtaoJiang/LayerCAM-jittor](https://github.com/PengtaoJiang/LayerCAM-jittor)

# 前言：为什么这么做？
现有的由CNN架构产生类别激活图CAM的方法用的都是CNN最后的卷积层，最后的卷积层分辨率小，这样的类别激活图经常定位到目标物体的粗糙区域，限制了需要像素级精确物体定位任务的执行。所以，LayerCAM这篇文章的作者们就想到，可以利用CNN前面的卷积层，前面的特征层分辨率高，也许可以获得更细粒度的对象定位信息。然后把由前面的特征层产生的类别激活图和后面的特征层产生的类别激活图都用上，集成在一起，也许就可以更好地突出与对象相关的像素，事实证明确实是这样。
所以为什么会有这篇文章，为什么要这么做呢？作者在前面介绍的部分说了：Due to the low spatial resolution of the output from the final convolution layer, the resulting class activation maps can only locate coarse object regions.也就是最后特征层的低分辨率导致了类别激活图粗糙，这会影响我们获得的语义分割图。所以，作者们想到，前面的特征层分辨率高，用改进的grad-cam方法去获得那些特征层的类别激活图(细粒度的对象细节)，再和最后特征层的类别激活图(粗糙的对象区域)融合，就可以产生更好的CAM了。

---


# 一、这个方法怎么work的？
这个方法怎么work的，作者就主要做了一个改变，就是改变grad-cam中正向传播激活值和反向传播梯度值的相乘那一部分，为什么要改变，因为作者用现成的grad-cam方法为浅层的feature map产生类别激活图时，发现很糟糕，从下图中可以看出，
![layercam
](https://img-blog.csdnimg.cn/ee987c33085942a1bd73bda2a2a458b2.png#pic_center)

用grad-cam从前面的feature map产生的类别激活图中，具有强值的点散布在整个图像周围，定位变得更差。这是由于grad-cam本身的产生CAM的机制：grad-cam利用特征图的平均梯度来表示其对目标类别的重要性，grad-cam产生CAM过程如下图所示。
![grad-cam](https://img-blog.csdnimg.cn/74aab078e7394d55b17a1fa0fc407479.png#pic_center)
简要介绍一下：我们只看最上面的Image Classification分支，首先网络进行正向传播，得到特征层A(最后一个卷积层的输出)和网络预测值y(注意，这里是softmax激活之前的数值)。假设我们想看一下网络针对Tiger Cat这个类别的感兴趣区域，假设网络针对Tiger Cat类别的预测值为$y^{c}$。接着对$y^{c}$进行反向传播，能够得到反传回特征层A的梯度信息 Á ，那么Á 就是$y^{c}$对A求得的偏导，换句话说，Á代表A中每个元素对$y^{c}$的贡献，贡献越大网络就认为越重要。然后对Á在$w$,$h$上求均值就能得到针对A每个通道的重要程度(对于类别$c$而言的)。最后进行简单的加权求和再通过ReLU就能得到文中所说的Grad-CAM。
关于Grad-CAM总结下来就是下面这个公式：

$$
L_{\mathrm{Grad}-\mathrm{CAM}}^{c}=\operatorname{ReLU}\left(\sum_{k} \alpha_{k}^{c} A^{k}\right)
$$

· $A$代表某个特征层，在grad-cam论文中一般指的是最后一个卷积层输出的特征层
· $k$代表特征层$A$中第$k$个通道(channel)
· $c$代表类别$c$
· $A^{k}$代表特征层$A$中通道$k$的数据
· $\alpha_{k}^{c}$代表针对$A^{k}$的权重

$\alpha_{k}^{c}$的计算公式如下：

$$
\alpha_{k}^{c}=\frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^{c}}{\partial A_{i j}^{k}}
$$

· $y^{c}$代表网络针对类别$c$预测的分数(score)，注意这里没有通过softmax激活
· $A_{i j}^{k}$代表特征层A在通道$k$中，坐标为$i$,$j$位置处的数据
· $Z$等于特征层的宽度×高度
通过计算公式可知$\alpha_{k}^{c}$就是通过预测类别$c$的预测分数$y^{c}$进行反向传播，然后利用反传到特征层A上的梯度信息计算特征层A每个通道$k$的重要程度。接着通过$\alpha$对特征层A每个通道的数据进行加权求和，最后通过ReLU激活函数得到Grad-CAM(论文中说使用ReLU是为了过滤掉Negative pixels)，而Negative pixels很可能是归属于其他类别的pixles。最后，经过简单的上采样就得到可视化结果了。 
浅层特征层用正常grad-cam方法不好的原因在于$\alpha_{k}^{c}$是将特征层A的每个通道$k$的梯度求平均，作为每个通道$k$的权重，也就是说grad-cam只考虑捕获每个特征图的全局信息，丢失了局部差异，看下图：
![grad-cam方差](https://img-blog.csdnimg.cn/690f356b2aac46469a274669b5f78992.png#pic_center)
可以看到，在浅层，对于大多数特征层，方差非常大，全局权重是不能代表特征图上不同位置对某一类别的重要性的，所以，浅层直接带入grad-cam得到的类别激活图不行。
为此作者做了一个改变，不再用全局权重，而是用像素级权重，也就是对应于特征图上的一个位置，如果梯度为正，则用这个正梯度作为权重，如果梯度为负，则权重为零(也就是用个ReLU)。数学公式比较好理解(不解释)，如下：
$$
w_{i j}^{k c}=\operatorname{relu}\left(g_{i j}^{k c}\right)
$$

$$
\hat{A}_{i j}^{k}=w_{i j}^{k c} \cdot A_{i j}^{k}
$$

$$
M^{c}=\operatorname{ReLU}\left(\sum_{k} \hat{A}^{k}\right)
$$
总结就是：将feature map每个位置的激活值乘以一个梯度权重。
需要注意的是，在文章的experiments中，作者有提到实现的细节，这对于最后算法的结果很重要：
(1)对于vgg16的前几个特征层，产生的CAM要用GraphCut方法产生连接段(官网代码貌似没有)。
(2)对于vgg16的前3个stage的特征层，产生的CAM值小，不能直接和后面的stage相加，所以用下面的式子scale一下：
$$
\hat{M}^{c}=\tanh \left(\frac{\gamma * M^{c}}{\max \left(M^{c}\right)}\right)
$$
(3)最后融合的时候，先归一化，然后用最大值操作融合不同layer的CAM。

---

# 二、结果可视化

参考官方代码，以及另一份代码，我们稍作修改，得到如下图的可视化结果。
每一行从左到右依次是原图，vgg stage1，vgg stage2,  vgg stage3,  vgg stage4,  vgg stage5, 融合图 
![在这里插入图片描述](https://img-blog.csdnimg.cn/1bafe55d30a3471295866d1b0d5285e0.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/6762e9237c5e4a5aa795e1f53fa8d9e3.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/7d94870a57494c07bcd667ca72dc31ee.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/9036191796c142ffb04504d4b1899b2b.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/d73d2e251bd54b9b955c1c69a221b421.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/d121ba74f452453586485350cb226649.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/b9b153b4a47d401bb8a9773f6fe29adf.png#pic_center)
可视化这一部分没有卡阈值。





---

# 三、总结
这篇文章可学习借鉴的地方有：
(1)作者有想到利用浅层的卷积层得到细粒度类别激活图
(2)用grad-cam方法得到浅层类别激活图效果不好后，作者的分析方式值得学习
(3)作者在融合过程使用的一些trick
(4)浅层的轮廓有点好
这篇文章不太好的地方在于对浅层的CAM没有设计很好地使用方法，而且，浅层的噪点太多了，有很多都分布在其他类别的物体上了，虽然本文的定位(localization)性能提上去了，但是对于实例分割任务，还要改动。
