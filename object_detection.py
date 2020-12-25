import os 
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# loading trained model
model = load_model('weights/mobilenet_modele.h5')

# video and boxes directory 
videos_dir = 'VIDEOS/'
boxes_dir =  'GT/'

# default video displayed if no arguments are passed
video_path = os.listdir(videos_dir)[0] 
gt_path = os.listdir(boxes_dir)[0]

# choose name and gt of displayed video
ap = argparse.ArgumentParser()
ap.add_argument("--video_path", default = videos_dir + str(video_path))
ap.add_argument("--gt_path", default = boxes_dir + str(gt_path))
video_path = ap.parse_args().video_path
gt_path = ap.parse_args().gt_path

# open txt file of boxes 
box_name = gt_path
box = open(box_name)

# class names dictionnary 
objects_dict = {0 : 'Bowl', 1 : 'CanOfCocaCola', 2 : 'MilkBottle', 3 : 'Rice', 4 : 'Sugar'}

# boxes dictionnary for each frame of the video
boxes = {}
for position, line in enumerate(box) : 
    if len(line.split()) == 6 :
        x = int(line.split()[2])
        y = int(line.split()[3])
        w = int(line.split()[4])
        h = int(line.split()[5])
        boxes[position] = (x, y, w, h)

# read video
position = 1 # first frame 
cap = cv2.VideoCapture(video_path)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False : 
        break 
    if position in boxes :
        # get position of object and draw rectangle
        x, y , w, h = boxes[position]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # cropping the region of interest (detected object)
        roi = frame[y:y + h, x:x + w]
        cropped_img =  np.expand_dims(np.expand_dims(cv2.resize(roi, (224, 224)), -1), 0)

        # prediction on the detected object 
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, objects_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break 
    position += 1

cap.release()