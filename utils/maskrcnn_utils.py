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
import time
import yaml
import os

def get_model_instance_segmentation(num_classes, weights_path):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    model.load_state_dict(torch.load( weights_path, map_location=torch.device('cuda')))
    model.to(torch.device('cuda'))
    model.eval()

    return model

def get_predictions(img, model, device, threshold = 0.6):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    transforms = T.Compose([T.ToTensor()])
    img = transforms(img).to(device)
    pred = model([img])
    #pred score
    pred_score = pred[0]['scores'].cpu().detach().numpy()
    end = len([x for x in list(pred_score) if x > threshold])
    #boxes N*4
    boxes = pred[0]['boxes'].cpu().detach().numpy().astype(int)
    #labels
    labels = pred[0]['labels'].cpu().detach().numpy()
    #masks
    masks = (pred[0]['masks'] >= 0.5).cpu().detach().squeeze(axis = 1).numpy()
    
    #taking only good scores
    boxes = boxes[:end]
    labels = labels[:end]
    masks = masks[:end]
    return boxes, labels, masks

def colour_masks(image, color):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = list(color)
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_image(img, objects, intruders,points):
    blue = (255, 0, 0)
    green = (0, 255, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (0, 0, 255)
    cyan = (255, 255, 0)
    for (objectID, obj) in objects.items():
        text = "person {}".format(objectID)
        if objectID in intruders:
            color = red
            mask_color = red
            cv2.putText(img, "Tailgating Detected", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 3,black,4,cv2.LINE_AA)
        else:
            color = green
            mask_color = cyan
        rgb_mask = colour_masks(obj['mask'], mask_color)
        img = cv2.addWeighted(img, 1.0, rgb_mask,0.2, 0)
        cv2.rectangle(img, (obj['box'][0], obj['box'][1]), (obj['box'][2], obj['box'][3]), color,3)
        cv2.putText(img, text, (obj['box'][0], obj['box'][1]-3), cv2.FONT_HERSHEY_SIMPLEX, 1,black,2,cv2.LINE_AA)
        ptx = (obj['box'][0] + obj['box'][2])//2
        pty = int(obj['box'][3])
        cv2.circle(img, (ptx, pty), 4, black, -1)
        img = cv2.line(img,points[0],points[1],blue,3) #End points of line
    return img