import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self,gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self,input,target):

        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input soze ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        print("max_val:",max_val)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()

'''Y_true = np.array([[1]])
Y_pred = np.array([[0.5]],dtype=np.float32)
fc = FocalLoss()
print(fc.forward(torch.from_numpy(Y_pred),torch.from_numpy(Y_true.astype(np.float32))))'''
'''import cv2
import matplotlib.pyplot as plt

# 使用matplotlib展示多张图片
def matplotlib_multi_pic1():
    for i in range(9):
        img = cv2.imread('apple.jpg')
        title="title"+str(i+1)
        #行，列，索引
        plt.subplot(3,3,i+1)
        plt.imshow(img)
        plt.title(title,fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.show()
matplotlib_multi_pic1()'''





