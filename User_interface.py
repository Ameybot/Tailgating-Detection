import PySimpleGUI as sg
import cv2
import numpy as np
from pathlib import Path
import torch
import maskRCNN_people_tracker
import yaml
import SORT_people_tracker
from utils.models import *
from utils.sort import *
import warnings
warnings. filterwarnings('ignore') 

from utils import datasets, mailer, maskrcnn_utils, models, parse_config, proximity_tracker, sort, utils

sg.theme("LightGreen")

CONFIG={}
with open('config\config.yaml') as f:
    global_val=yaml.load(f)
    CONFIG['config_path']=global_val['models'].get('yolov3_cfg')
    CONFIG['weights_path']=global_val['models'].get('yolov3_weights')
    CONFIG['weights'] = global_val['models'].get('mask_rcnn_weights')
    CONFIG['class_path']=global_val['models'].get('yolov3_class')
    CONFIG['img_size']=416
    CONFIG['conf_thres']=0.8
    CONFIG['nms_thres']=0.4
    CONFIG['video'] = 'tailgate_3.mp4' #Replace with 0 or IP Address of CCTV for Real-Time Detection
    CONFIG['sender_email_address'] = global_val['email'].get('sender_email_address')
    CONFIG['email_password'] = global_val['email'].get('email_password')
    CONFIG['receiver_email_address'] = global_val['email'].get('receiver_email_address')
    CONFIG['intruder_image_folder'] = global_val['email'].get('intruder_image_folder')
    CONFIG["device"]=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
f.close()

model = Darknet(CONFIG['config_path'], img_size=CONFIG['img_size'])
model.load_weights(CONFIG['weights_path'])
model.to(CONFIG['device'])
model.eval()
classes = utils.load_classes(CONFIG['class_path'])
model_ = maskrcnn_utils.get_model_instance_segmentation(2,  CONFIG['weights'])
    # Define the window layout
layout = [
    [sg.Text("Tailgating Detector", size=(60, 1), justification="center")],
    [sg.Image(filename="", key="-IMAGE-")],
    [sg.Output(size=(80, 5))],    
    [sg.Radio("MaskRCNN+Centroid(Horizontal) tracker", "Radio", True, size=(30, 1), key='rishon')],
    [sg.Radio("YOLOv3+SORT Tracker", "Radio", True, size=(30, 1) , key='aishik')],
    # [sg.Radio("MaskRCNN+Centroid(Vertical) tracker", "Radio", True, size=(30, 1), key='amey')],
    # [sg.Listbox(['tailgate_1.mp4', 'tailgate_2.mp4', 'tailgate_3.mp4', 'tailgate_4.mp4', 'tailgate_5.mp4'], size=(20,4), enable_events=False, key='videos')],
    [sg.Button("Exit", size=(10, 1))]]

window = sg.Window("Tail Gating Detection- Team Classy-fiers", layout, location=(800, 400))

cap = cv2.VideoCapture('tailgate_5.mp4 ')
frames=0
mot_tracker = Sort() 
prev_frame = 0
glob_cross = 0
intruder = 0
ct = proximity_tracker.CentroidTracker()
signal = 1 #signal that card is opened
allowed = []
intruders = set()
done = 0
frames=0

while True:
    event, values = window.read(timeout=20)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    ret, frame = cap.read()
    frames+=1
    if (frames==1):
        r = cv2.selectROI(frame)
    if values["aishik"]:        
        frame, prev_frame, glob_cross, intruder=SORT_people_tracker.sort_detection_tracking(frame, r, classes, model, CONFIG, mot_tracker, prev_frame, glob_cross, intruder)
    elif values["rishon"]:
        frame, allowed, ct, signal, intruders, done=maskRCNN_people_tracker.obj_detection_vid_api(frame, model_, r, CONFIG['device'], CONFIG, allowed, ct,signal, intruders,done)
 

    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
    window["-IMAGE-"].update(data=imgbytes)

window.close()


