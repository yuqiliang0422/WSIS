from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from json_inference import coco_encode, coco_inst_seg_eval

from PIL import Image
import matplotlib.pyplot as plt
import scipy

import numpy as np
import torch
import json

annFile = "/mass/wsk/dataset/coco2014/annotations/instances_val2017.json"
# result_file = "/mass/lyq/lyq/PycharmProjects/prm/tools/prm_tools/Outputs/coco2014/prm_vgg_v3/vit2_json"
result_file = "/mass/lyq/lyq/PycharmProjects/prm/tools/prm_tools/Outputs/voc2012sbd/prm_vgg_voc_v3/cocolast10_json"


cocoGT = COCO(annFile)
imgIds = sorted(cocoGT.getImgIds())

annType = ['segm','bbox','keypoints']
annType = annType[0]
cocoGt = COCO(annFile)
cocoDt = cocoGt.loadRes(result_file)
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

mAP, cls_ap = coco_inst_seg_eval(annFile, result_file)
print('Performance(COCOAPI):')
for k, v in mAP.items():
    print('mAP@%s: %.1f' % (k, 100 * v))
print("all_class",cls_ap)



