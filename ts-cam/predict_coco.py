import os
import json
import sys

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.misc import imresize
import torch.nn.functional as F
from collections import OrderedDict
from coco_dataset import coco_nummap_id
from scipy.misc import imresize

# -*- coding: utf-8 -*-
from pycocotools import mask as COCOMask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from six.moves import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image
from math import pow,sqrt
import numpy as np
from pycocotools import mask as maskUtils

VOC_COLORMAP = [[85, 26, 139], [238, 238, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]



def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP).cuda()
    X = torch.tensor(pred).long()
    return colormap[X, :]

def scipy_misc_imresize(arr,size,interp='bilibear',mode=None):
    im = Image.fromarray(np.uint8(arr),mode=mode)
    ts = type(size)
    if np.issubdtype(ts,np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size)*percent).astype(int))
    elif np.issubdtype(type(size),np.floating):
        size = tuple((np.array(im.size)*size).astype(int))
    else:
        size = (size[1],size[0])
    func = {'nearest':0,'lanczos':1,'biliear':2,'bicubic':3,'cubic':3}
    imnew = im.resize(size,resample=func[interp])    # 调用PIL库中的resize函数
    return np.array(imnew)

coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

'''coco_id_name_map = {1:'aeroplane',2:'bicycle',3:'bird',4:'boat',5:'bottle',6:'bus',7:'car',8:'cat',9:'chair',10:'cow',
                  11:'diningtable',12:'dog',13:'horse',14:'motorbike',15:'person',16:'pottedplant',17:'sheep',18:'sofa',
                  19:'train',20:'tvmonitor'}'''

from vit_model import vit_base_patch16_224_in21k as create_model
from vit_module import peak_stimulation,_median_filter

# root = '/mass/wsk/dataset/VOCdevkit/VOC2012/JPEGImages/'
# label_file = '/mass/wsk/dataset/VOCdevkit/VOC2012/annotations/voc_2012_val.json'
root = '/mass/wsk/dataset/coco2014/val2017/'
label_file = "/mass/wsk/dataset/coco2014/annotations/instances_val2017.json"
cocoGT = COCO(label_file)
bbox_props = dict(facecolor='white', edgecolor='none', alpha=0.6, zorder=2)

# save_dir = "/mass/lyq/lyq/PycharmProjects/prm_cam/try1"



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # { "0": "daisy", "1": "dandelion", "2": "roses", "3": "sunflowers", "4": "tulips}
    data_transform = transforms.Compose([
         transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load image
    # img_path = "/mass/lyq/lyq/PycharmProjects/prm_cam/vision_transformer/data/CUB_200_2011/images/024.Red_faced_Cormorant/Red_Faced_Cormorant_0009_796314.jpg"
    # img_path = "/mass/wsk/dataset/coco2014/val2017/vision_transformer/data/CUB_200_2011/images/024.Red_faced_Cormorant/Red_Faced_Cormorant_0009_796314.jpg"
    # img_path = "/mass/wsk/dataset/coco2014/val2017/000000181753.jpg"
    imgIds = sorted(cocoGT.getImgIds())
    for index in range(60,61):  # index = 5

        # img_id = [imgIds[index]]
        img_id = 437514
        gt_ann_ids = cocoGT.getAnnIds(imgIds=img_id)
        anns = cocoGT.loadAnns(gt_ann_ids)

        path = cocoGT.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(root, path)).convert('RGB')
        height = img.size[1]
        width = img.size[0]
        x = data_transform(img)
        print(x.size())
        x = x.unsqueeze(0).to(device)
        img_after = scipy_misc_imresize(img, (224, 224), interp='bicubic')
        img_after = Image.fromarray(img_after.astype('uint8')).convert('RGB')

        # [N, C, H, W]

        # create model
        model = create_model(num_classes=80, has_logits=False)
        print(model)
        # load model weights
        model_weight_path = "/mass/lyq/lyq/PycharmProjects/prm_cam/vision_transformer/weights_coco/model_loss-9.pth"
        new_state_dict = OrderedDict()
        state_dict = torch.load(model_weight_path)
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。

        model.load_state_dict(new_state_dict)
        # model.inference().to(device)
        model = model.eval().to(device)
        # f, axarr = plt.subplots(3, 10, figsize=(2 * 4, 4))
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.01)
        with torch.no_grad():
            predict, cams_re, attn_weights, tscams_ar, cams, joint_attns, p, feature_map, cams_attention = model(x)
        # with torch.autograd.set_detect_anomaly(True):
            # predict, tscams_ar, valid_peak_list, peak_response_maps, peak_score = model(x)
            # print("peak_response_maps.shape:",peak_response_maps.shape)
            # print("peak_response_maps:",peak_response_maps)
            # valid_peak_list = valid_peak_list.cpu().numpy()
            # leng = len(valid_peak_list)
            cams_re = cams_re.squeeze(0).squeeze(0).cpu().numpy()
            cams_re = cv2.resize(cams_re / cams_re.max(), (14, 14))
            # cams_re = (cams_re[...,np.newaxis] * img_after).astype("uint8")
            # f, axarr = plt.subplots(1, 1, figsize=(2 * 4, 4))
            # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.01)


            predict = torch.squeeze(predict).cpu()
            predicts = predict.numpy()
            num = []
            print(predicts)

            for q in range(len(predicts)):
                if predicts[q] > 0:
                    num.append(q)
            length = len(num)
            print(num)
            predict = torch.softmax(predict, dim=0)
            predict_cla = torch.argmax(predict).numpy()

            f, axarr = plt.subplots(8, 10, figsize=(8, 10))
            plt.subplots_adjust(left=0, bottom=0, right=0.9, top=0.6, wspace=0.001, hspace=0.01)
            # feature_map = feature_map.squeeze(0).cpu()
            tscams_ar = tscams_ar.squeeze(0).cpu()
            for i in range(8):
                for j in range(10):
                    axarr[i, j].imshow(tscams_ar[10*i+j])
                    axarr[i, j].axis('off')
            # plt.imshow(cams_re)
            # plt.imshow(cams_re,cmap = "viridis",alpha=0.8)
            # plt.axis('off')
            plt.savefig("tscams_ar.jpg", dpi=300)

            f, axarr = plt.subplots(1, 1, figsize=(8, 8))
            polygons = []
            color = []
            rectangles = []
            rectangles_color = []
            axarr.imshow(img_after)
            axarr.axis('off')
            for ann in anns:  # ann=anns[0]
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            for i in range(len(seg) // 2):
                                seg[2 * i] = 224 * seg[2 * i] / width
                                seg[2 * i + 1] = 224 * seg[2 * i + 1] / height
                            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                            rectangles_color.append(c)
                            text_color = c
                    else:
                        # mask
                        t = cocoGT.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        m = m.squeeze()
                        m_after = imresize(m, (224, 224), interp='nearest')
                        img1 = np.ones((m_after.shape[0], m_after.shape[1], 3))
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0, 166.0, 101.0]) / 255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img1[:, :, i] = color_mask[i]
                        axarr.imshow(np.dstack((img1, m_after * 0.6)))
                        rectangles_color.append(color_mask)
                        text_color = color_mask
                    # box and label:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    bbox_x = 224 * bbox_x / width
                    bbox_y = 224 * bbox_y / height
                    bbox_w = 224 * bbox_w / width
                    bbox_h = 224 * bbox_h / height
                    poly = [[bbox_x, bbox_y], \
                            [bbox_x, bbox_y + bbox_h], \
                            [bbox_x + bbox_w, bbox_y + bbox_h], \
                            [bbox_x + bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4, 2))
                    rectangles.append(Polygon(np_poly))
                    # [0.0, 1.0, 0.0]) # green
                    if 'score' in ann:
                        ax.text(bbox_x, bbox_y - 2, \
                                '%s: %.2f' % (cocoGT.loadCats(ann['category_id'])[0]['name'], ann['score']), \
                                color='black', fontsize=15, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8,
                                                                      zorder=2))  # text_color  bbox_props)
                    else:
                        axarr.text(bbox_x, bbox_y - 2, \
                                         '%s' % (cocoGT.loadCats(ann['category_id'])[0]['name']), \
                                         color='black', fontsize=3,
                                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8,
                                                   zorder=2))
                        # color=text_color, fontsize=10, bbox=bbox_props)
                else:
                    color_mask = np.array([2.0, 166.0, 101.0]) / 255
                    rectangles_color.append(color_mask)
                    text_color = color_mask
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    bbox_x = 224 * bbox_x / width
                    bbox_y = 224 * bbox_y / height
                    bbox_w = 224 * bbox_w / width
                    bbox_h = 224 * bbox_h / height
                    poly = [[bbox_x, bbox_y], \
                            [bbox_x, bbox_y + bbox_h], \
                            [bbox_x + bbox_w, bbox_y + bbox_h], \
                            [bbox_x + bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4, 2))
                    rectangles.append(Polygon(np_poly))
                    # # [0.0, 1.0, 0.0]) # green
                    if 'score' in ann:
                        axarr.text(bbox_x, bbox_y - 2, \
                                         '%s: %.2f' % (cocoGT.loadCats(ann['category_id'])[0]['name'], ann['score']), \
                                         color='black', fontsize=5,
                                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8,
                                                   zorder=2))  # text_color  bbox_props)
                    else:
                        axarr.text(bbox_x, bbox_y - 2, \
                                         '%s' % (cocoGT.loadCats(ann['category_id'])[0]['name']), \
                                         color='black', fontsize=5,
                                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8,
                                                   zorder=2))

            p = PatchCollection(polygons, facecolor=color, linewidths=1, alpha=0.5)
            axarr.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=1)
            axarr.add_collection(p)
            p = PatchCollection(rectangles, facecolor='none', edgecolors=rectangles_color, linewidths=1)
            axarr.add_collection(p)


            '''for p in range(leng):
                axarr[p+2].imshow(peak_response_maps[p].cpu().numpy(),cmap="gray")
                axarr[p+2].axis('off')
                axarr[p+2].text(bbox_x, bbox_y - 2, \
                                 '%s' % (str(coco_id_name_map[valid_peak_list[p][1]+1])), \
                                 color='black', fontsize=5,
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8,
                                           zorder=2))'''
            plt.savefig("boat.jpg", dpi=300)





        # class_response_maps = F.upsample(tscams_ar, scale_factor=32, mode='bilinear', align_corners=True)
        '''class_response_maps = tscams_ar[0, num[0], :, :].detach().cpu().numpy()
        class_response_maps = cv2.resize(class_response_maps, img_after.size)
        min_v, max_v = class_response_maps.min(), class_response_maps.max()
        class_response_maps = (class_response_maps - min_v) / (max_v - min_v) * 255
        class_response_maps1 = tscams_ar[0, num[0], :, :].detach().cpu().numpy()
        class_response_maps1 = cv2.resize(class_response_maps1, img_after.size)
        min_v1, max_v1 = class_response_maps1.min(), class_response_maps1.max()
        class_response_maps1 = (class_response_maps1 - min_v1) / (max_v1 - min_v1) * 255
        axarr[0].imshow(img_after)
        axarr[0].axis('off')
        axarr[1].imshow(peak1,cmap=plt.cm.jet)
        axarr[1].axis('off')
        axarr[2].imshow(peak2,cmap=plt.cm.jet)
        axarr[2].axis('off')
        # axarr[0, 4].imshow(peak3)
        axarr[3].imshow(class_response_maps)
        axarr[3].axis('off')
        axarr[4].imshow(class_response_maps1)
        axarr[4].axis('off')
        plt.savefig("1.jpg", dpi=300, bbox_inches='tight')
        plt.close('all')'''

        '''for i in range(len(num)):
            print(coco_id_name_map[coco_nummap_id[num[i]]])
            cam_pred_ar = tscams_ar[0, num[i], :, :].detach().cpu().numpy()
            mask_pred_ar = cv2.resize(cam_pred_ar, img_after.size)
            mask_ar_min_v, mask_ar_max_v = mask_pred_ar.min(), mask_pred_ar.max()
            mask_pred_ar = (mask_pred_ar - mask_ar_min_v) / (mask_ar_max_v - mask_ar_min_v)

            # mask_pred_ar = cv2.resize(mask_pred_ar, img_after.size)
            class_response_maps = torch.tensor(mask_pred_ar).unsqueeze(0).unsqueeze(0)
            peak_list = peak_stimulation(class_response_maps, return_aggregation=False, win_size=3, peak_filter=_median_filter)
            mask_pred_ar = mask_pred_ar * 255
            print(len(peak_list))
            peak_list_new = []
            for j in range(len(peak_list)):
                if class_response_maps[peak_list[j, 0], peak_list[j, 1], peak_list[j, 2], peak_list[j, 3]] > 0.8:
                    peak_list_new.append(peak_list[j])
            print(peak_list_new)
            peak_list_new = BubbleSort(mask_pred_ar, peak_list_new)
            matrix = torch.zeros((224,224))
            x_main = peak_list_new[0][2]
            y_main = peak_list_new[0][3]
            color = mask_pred_ar[peak_list_new[0][2]][peak_list_new[0][3]]

            print("x_main,y_main:",x_main,y_main)
            for m in range(224):
                for n in range(224):
                    matrix[m][n] = sqrt(pow((x_main - m),2) + pow((y_main - n),2) + pow((color - mask_pred_ar[m][n]),2) * 2 )'''


if __name__ == '__main__':
    main()



