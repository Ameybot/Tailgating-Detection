import cv2
import yaml
from utils.sort import *
from utils.models import *
from utils import utils, datasets, mailer, parse_config
from imutils.video import FPS

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image



def detect_image(img, img_size,conf_thres, nms_thres, model, device):
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = torch.tensor(image_tensor, dtype=torch.float).to(device)
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

def check_cross_difficult(centroids,pt_1,pt_2,pt_3,pt_4):
    count = 0 # Isko bhi argument ki tarah lo. 
    
    a = pt_2[1] - pt_1[1]
    b = pt_1[0] - pt_2[0]
    c = pt_4[1] - pt_3[1]
    d = pt_3[0] - pt_4[0]
    
    e = pt_1[1] * (pt_2[0] - pt_1[0]) + pt_1[0]*(pt_1[1] - pt_2[1])
    f = pt_3[1] * (pt_4[0] - pt_3[0]) + pt_3[0]*(pt_3[1] - pt_4[1])
    for i in centroids:
        if e*(a*i[0] + b*i[1]+e) < 0 and f*(c*i[0] + d*i[1]+f) > 0: #Greater sign if same side as origin
            count += 1

    return count

def sort_detection_tracking(frame, r, classes, model, CONFIG, mot_tracker, prev_frame, glob_cross, intruder):
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
    starttime = time.time()
    if (False): #Swipe Signal
        glob_cross = 0
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    r=np.array(r).astype(int)
    model_frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.rectangle(frame, ( int(r[0]),int(r[1])), (int(r[0]+r[2]),int(r[1]+r[3])), (0,0,0), 4)

    x_offset=r[0]
    y_offset=r[1]

    pilimg = Image.fromarray(model_frame)
    detections = detect_image(pilimg, CONFIG['img_size'],CONFIG['conf_thres'], CONFIG['nms_thres'], model, CONFIG['device'])

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
       
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (CONFIG['img_size'] / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (CONFIG['img_size'] / max(img.shape))
    unpad_h = CONFIG['img_size'] - pad_y
    unpad_w = CONFIG['img_size'] - pad_x

    cent = []
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())
        # print(tracked_objects)
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]
            if(cls=='person'):
                cv2.rectangle(frame, (x1+x_offset, y1+y_offset), (x1+x_offset+box_w, y1+y_offset+box_h), color, 4)
                cv2.rectangle(frame, (x1+x_offset, y1+y_offset-35), (x1+x_offset+len(cls)*19+80, y1+y_offset), color, -1)
                cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1+x_offset, y1+y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                cent.append(((2*x1+2*x_offset+box_w)/2,(2*y1+2*y_offset+box_h)/2))
                cv2.circle(frame, np.array(cent[-1]).astype(int), radius=5, color=(255, 0, 0), thickness=-1)
    
    cv2.line(frame,(r[0], r[1]), (r[0], r[1]+r[3]),(0,0,255),3) #End points of line
    cv2.line(frame,(r[0]+r[2],r[1]),(r[0]+r[2],r[1]+r[3]),(0,0,255),3)    #End points of line

    cross = check_cross_difficult(cent,(r[0], r[1]), (r[0], r[1]+r[3]),(r[0]+r[2],r[1]),(r[0]+r[2],r[1]+r[3]))
    if(prev_frame!=cross):
        prev_frame = cross
        glob_cross += prev_frame
        print('CROSS: ',glob_cross)
        if cross>0 and glob_cross>1:
            intruder += 1
            cv2.imwrite('Intruders/Intruder_{}.jpg'.format(intruder),frame)
            mailer.mail("Aishik Rakshit", CONFIG['sender_email_address'],  CONFIG['receiver_email_address'],  CONFIG['intruder_image_folder'],  CONFIG['email_password'])
    else:
        print('CROSS: ',glob_cross)        
    if  glob_cross>1:
        frame = cv2.putText(frame, 'Tailgating Detected', (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 256), 3, cv2.LINE_AA)
    else:
        frame = cv2.putText(frame, 'No Tailgating Detected', (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 256, 0), 3, cv2.LINE_AA)
    frame=cv2.resize(frame, (780, 540),interpolation = cv2.INTER_NEAREST)
    return frame,prev_frame, glob_cross, intruder

