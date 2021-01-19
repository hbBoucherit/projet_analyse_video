# imports
import os 
import argparse
import cv2
from random import randint
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess

aspect_ratios = [[1,1], [0.5,0.5], [1.5, 1.5], [2.5, 2.5]]
steps = [0.5, 1, 1.5]

# loading trained vgg model
# model = load_model('weights/vgg16_based__9.h5')
model = load_model('projet_analyse_video/weights/vgg16_based__9.h5')

# class names dictionnary 
class_names = {0 : 'Bowl', 1 : 'CanOfCocaCola', 2 : 'MilkBottle', 3 : 'Rice', 4 : 'Sugar'}

def get_candidate_patches(frame, candidate_boxes):
    patches = []
    for box in candidate_boxes:
        x, y, w, h = [int(t) for t in box]
        crop = frame[y:y + h, x:x + w] # crop
        patches.append(cv2.resize(crop, (224, 224))) # resize

    return np.stack(patches, axis=0) # return stacked patches


class_indexes = {'Bowl': 0, 'CanOfCocaCola' : 1, 'MilkBottle' : 2, 'Rice' : 3, 'Sugar' : 4}

def read_annotation_file(file_path):
    # open txt file of boxes 

    class_name = Path(file_path).name.split('Place')[0]

    print('class_name:', class_name)
    label_index = class_indexes.get(class_name)

    assert label_index is not None, ValueError(f'Could not get label index from file {file_path}')
    
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        boxes = { idx: [int(x) for x in row.split()[2:6]] for idx, row in enumerate(lines) if len(row.split()) == 6}
    # print()

    return label_index, boxes

def get_candidate_boxes(box, steps, aspect_ratios, im_width, im_height):

    x, y, w, h = box

    boxes = []
    
    for ratio in aspect_ratios:
        for step in steps: 
            boxes += [[x + step*w , y , ratio[0] * w, ratio[1] * h],
                      [x - step*w , y , ratio[0] * w, ratio[1] * h],
                      [x , y + step*h, ratio[0] * w, ratio[1] * h],
                      [x , y - step*h, ratio[0] * w, ratio[1] * h]]

    boxes = np.array(boxes).astype(int)

    return clip_boxes(boxes, im_width, im_height)


def clip_boxes(boxes, im_width, im_height):

    xyxy_boxes = xywh2xyxy(boxes)

    # Clamp boxes outside of the bounding box
    cliped_boxes = np.zeros_like(boxes)
    cliped_boxes[:,0] =  np.clip(xyxy_boxes[:, 0], a_min=0, a_max=im_width-1)  # x
    cliped_boxes[:,1] =  np.clip(xyxy_boxes[:, 1], a_min=0, a_max=im_height-1) # y
    cliped_boxes[:,2] =  np.clip(xyxy_boxes[:, 2], a_min=0, a_max=im_width-1)  # x + w
    cliped_boxes[:,3] =  np.clip(xyxy_boxes[:, 3], a_min=0, a_max=im_height-1) # y + h

    new_xywh_boxes = xyxy2xywh(cliped_boxes)
    
    # a box is unclamped if all its new coords are equal to the original unclamped coords
    diff_boxes = abs(new_xywh_boxes - boxes) <= 0 

    # boolean indices of boxes that the crop removed 40 pixels in each dimension at most
    inside = np.array([all(b) for b in diff_boxes])

    return boxes[inside]


def xyxy2xywh(boxes):
    """
    Convert xyxy coordinates to xywh
    """
    out = np.zeros_like(boxes)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x center
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y center
    out[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    out[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return out


def xywh2xyxy(boxes):
    """
    Convert xywh coordinates to xyxy
    """
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
    return out

# function that returns squarred bounding boxe based on the original bounding box
def squared_boxes(x,y,w,h):
    size_diff = abs((w-h)//2)
    if w >= h :
        h = w 
        y = y - size_diff # center the box
    else : 
        w = h
        x = x - size_diff # center the box
    return (x,y,w,h)

# video and boxes directory 
videos_dir = 'VIDEOS/'
boxes_dir =  'GT/'

# default video displayed if no arguments are passed
index_video = randint(0,len(os.listdir(videos_dir))-1)
video_path = os.listdir(videos_dir)[index_video] 
gt_path = os.listdir(boxes_dir)[index_video]

# choose name and gt of displayed video
ap = argparse.ArgumentParser()
ap.add_argument("--video_path", default = videos_dir + str(video_path))
ap.add_argument("--gt_path", default = boxes_dir + str(gt_path))
video_path = ap.parse_args().video_path
gt_path = ap.parse_args().gt_path

# read video
predictions = {}
labelIndex, bboxes = read_annotation_file(gt_path)
frame_number = 1 
cap = cv2.VideoCapture(video_path)
tracking = False

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False : 
        break

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if tracking or frame_number in bboxes : 
        if not tracking : 
            assert(frame_number in bboxes)
            current_bbox = bboxes[frame_number]
            tracking = True 
        else : 
            # get candidate patches
            candidate_bboxes = get_candidate_boxes(current_bbox, steps, aspect_ratios, fw, fh)
            candidate_bboxes_patches = get_candidate_patches(frame, candidate_bboxes)

            # select best candidate box
            candidate_bboxes_predictions = model.predict(preprocess(candidate_bboxes_patches))
            best_prediction_index = np.argmax(candidate_bboxes_predictions[:, labelIndex])
            best_bbox = candidate_bboxes[best_prediction_index]

            # Update current box
            current_bbox = best_bbox
            predictions[frame_number] = best_bbox

        x, y, w, h = current_bbox
        cv2.putText(frame, class_names[labelIndex], (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break 
    frame_number += 1

cap.release()

# anchors = [[1,1], [0.5, 0.5]]
# box = [x,y,h,w]

