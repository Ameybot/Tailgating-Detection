#Import Statements for Mailer Script
import smtplib
from email.message import EmailMessage
import imghdr
import os
from torch._C import CONV_BN_FUSION

#Other import statements
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
import random
import yaml
import time
import os
from utils import maskrcnn_utils, mailer
from utils import proximity_tracker


def find_crossed(objects,points):
    
    #returns a list of indexes who have crossed the line 
    pt_1 = points[0]
    pt_2 = points[1]
    a = pt_2[1] - pt_1[1]
    b = pt_1[0] - pt_2[0]
    c = pt_1[1] * (pt_2[0] - pt_1[0]) + pt_1[0]*(pt_1[1] - pt_2[1])
    crossed = []
    for (objectId, obj) in objects.items():
        ptx = (obj['box'][0] + obj['box'][2])/2
        pty = obj['box'][3]
        if c*(a*ptx + b*pty+c) > 0: #Greater sign if same side as origin
            crossed.append(objectId)

    return crossed

def obj_detection_vid_api(frame, model, r, device, CONFIG, allowed, ct,signal, intruders,done):
    # pt1=int(r[0])
    # pt2=int(2*r[1]+r[3])
    # pt3=int(r[0]+r[2])
    # pt4=int(2*r[1]+r[3])
    pt1=680
    pt2=580
    pt3=900
    pt4=650
    boxes, labels, masks = maskrcnn_utils.get_predictions(frame, model, device)
    objects = ct.update(boxes, labels, masks)
    if(signal == 1):
        check = find_crossed(objects,((pt1,pt2),(pt3,pt4)))
        if(len(check) > 0):
            allowed.extend(check)
            signal = 0
            print("Person {} granted access".format(allowed[-1]))
    crossed = find_crossed(objects,((pt1,pt2),(pt3,pt4)))
    
    for i in crossed:
        if i not in allowed:
            if done == 0:
                print("Tailgating detected!")
                print("Person {} is an intruder".format(i))
                masked = cv2.bitwise_and(frame, frame, mask=objects[i]['mask'].astype(np.uint8))
                cv2.imwrite('Intruders/intruder.jpg',masked)
                mailer.mail('Rishon Dsouza', CONFIG['sender_email_address'],  CONFIG['receiver_email_address'],  CONFIG['intruder_image_folder'],  CONFIG['email_password'])
                intruders.add(i)
                done = 1
    
    img = maskrcnn_utils.get_image(frame, objects, intruders,((pt1,pt2),(pt3,pt4)))
    img=cv2.resize(img, (780, 540),interpolation = cv2.INTER_NEAREST)
    return img, allowed, ct, signal, intruders, done