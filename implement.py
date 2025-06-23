import torch
import cv2
from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_300,VOC_512,COCO_300
from layers.functions import Detect,PriorBox
import os
from data.voc0712 import VOC_CLASSES
from data import VOCroot
from models.RFB_Net_vgg import build_net
import numpy as np
from utils.nms_wrapper import nms
# note: if you used our download scripts, this should be right

img_dim = 300
num_classes = 21
net = build_net('test', img_dim, num_classes)    # initialize detector
state_dict = torch.load('weights/Final_RFB_vgg_VOC.pth',map_location='cpu')
# create new OrderedDict that does not contain `module.`

from collections import OrderedDict

cfg = VOC_300
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:] # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
net.eval()
rgb_means = (104, 117, 123)
transform = BaseTransform(net.size, rgb_means, (2, 0, 1))

img = cv2.imread('fig/3512.jpg', cv2.IMREAD_COLOR)
x = transform(img).unsqueeze(0)
out = net(x)  # forward pass
detector = Detect(num_classes, 0, cfg)
boxes, scores = detector.forward(out, priors)
boxes = boxes[0]
scores = scores[0]
scale = torch.Tensor([img.shape[1], img.shape[0],
                      img.shape[1], img.shape[0]])
boxes *= scale
boxes = boxes.cpu().numpy()
scores = scores.cpu().numpy()
for i in range(boxes.shape[0]):
    j=np.max(scores[i,:])
    wh  = np.where(scores[i,:]==j)[0]
    name = VOC_CLASSES[int(wh)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    appear ='car'+' '+str(scores[i,int(j)])[0:5]
    img_or = img.copy()
    cv2.rectangle(img_or,(int(boxes[i,0]),int(boxes[i,1])),(int(boxes[i,2]),int(boxes[i,3])),(0,0,255),2)
    cv2.putText(img_or, appear, (int(boxes[i,0] + 5), int(boxes[i,1] + 10)), font, 0.5, (0, 0, 255), 1)
    cv2.imwrite('test_fig/' + str(i) + '.jpg', img_or)

