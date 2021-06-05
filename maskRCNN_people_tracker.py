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
        if c*(a*obj['centroid'][0] + b*obj['centroid'][1]+c) > 0: #Greater sign if same side as origin
            crossed.append(objectId)

    return crossed

def obj_detection_vid_api(vid_path, model, device, CONFIG):
    cap = cv2.VideoCapture(vid_path)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    ct = proximity_tracker.CentroidTracker()
    signal = 1 #signal that card is opened
    allowed = []
    intruders = set()
    done = 0
    frames=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames+=1
        key=cv2.waitKey(1) & 0xFF
        if (frames==1):
            r = cv2.selectROI(frame)
        pt1=int(r[0])
        pt2=int((2*r[1]+r[3])/2)
        pt3=int(r[0]+r[2])
        pt4=int((2*r[1]+r[3])/2)


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
        cv2.imshow('frame',img)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def execute():
    
    CONFIG = {}
    with open('config\config.yaml') as f:
        
        global_val = yaml.load(f)
        CONFIG['weights'] = global_val['models'].get('mask_rcnn_weights') #Path of Weights File
        CONFIG['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # CONFIG['line'] = ((680, 400),(900,500))
        CONFIG['video'] = 'tailgate_1.mp4' #Replace with 0 or IP Address of CCTV for Real-Time Detection
        CONFIG['sender_email_address'] = global_val['email'].get('sender_email_address')
        CONFIG['email_password'] = global_val['email'].get('email_password')
        CONFIG['receiver_email_address'] = global_val['email'].get('receiver_email_address')
        CONFIG['intruder_image_folder'] = global_val['email'].get('intruder_image_folder')

        model = maskrcnn_utils.get_model_instance_segmentation(2,  CONFIG['weights'])
        obj_detection_vid_api(CONFIG['video'], model, CONFIG['device'], CONFIG)

        
