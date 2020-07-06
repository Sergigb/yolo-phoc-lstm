import glob
import copy
import os
import math
import json
from math import ceil
import sys
import random

import numpy as np
import xml.etree.cElementTree as ET
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat

from utils import trans, iou


dataset_path = "../datasets/rrc-text-videos/ch3_train/"
annotations_paths = glob.glob(dataset_path + "*.xml")
#annotations_paths = ['../datasets/rrc-text-videos/ch3_train/Video_45_6_4_GT.xml']

random.shuffle(annotations_paths)

max_sequence_length = 50
att_size = 38
empty_gt = np.zeros((max_sequence_length, att_size, att_size))
num_descriptors = 361


if not os.path.isdir("gt"):
    os.mkdir("gt")

labels = dict()

for annotations_path in annotations_paths:
    voc_path = annotations_path.replace(".xml", "_voc.txt")
    video_path = annotations_path.replace('_GT.xml', '.mp4')
    video_name = video_path.split('/')[-1].replace('.mp4', '')

    cap = cv2.VideoCapture(video_path)
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    print(video_path)

    step_x = int(ceil(video_width/att_size))
    step_y = int(ceil(video_height/att_size))

    tree = ET.parse(annotations_path)
    root = tree.getroot()

    vocabulary = set()
    with open(voc_path) as f:
        lines = f.readlines()
    for line in lines:
        word = line.split(',')[-1]
        word = word.translate(trans).lower()
        vocabulary.add(word)

    for word in vocabulary:
        max_instances_frame = 0 # max number of instances of the same word in a single frame
        gt = copy.deepcopy(empty_gt)
        bbox_temp = []
        for i in range(max_sequence_length): bbox_temp.append([0, 0, 0, 0])

        frame_num = 0
        for frame in root:
            frame_num = int(frame.get("ID"))
            #print(frame_num)
            
            instances_frame = 0
            for object_ in frame:
                if object_.get("Transcription").translate(trans).lower() == word and max_instances_frame < 2:
                    instances_frame += 1

                    x_coords = []
                    y_coords = []
                    for point in object_.iter("Point"):
                        x_coords.append(int(point.get("x")))
                        y_coords.append(int(point.get("y")))

                    x1 = min(x_coords)
                    y1 = min(y_coords)
                    x2 = max(x_coords)
                    y2 = max(y_coords)

                    for i in range(0, int(video_width), step_x):
                        for j in range(0, int(video_height), step_y):
                            if iou([i, j, i+step_x, j+step_y], [x1, y1, x2, y2]):
                                gt[(frame_num-1) % max_sequence_length, int(j/step_y), int(i/step_x)] = 1  # i-j swapped because fuck me if I know
                    
                    bbox_temp[(frame_num-1) % max_sequence_length] = [x1, y1, x2-x1, y2-y1]
                    #1280.0 960.0

            if instances_frame > max_instances_frame:
                max_instances_frame = instances_frame

            if not (frame_num - 1) % max_sequence_length and frame_num != 1:
                if max_instances_frame == 1:  # only 1
                    fname = ("gt_" + video_name + "_" + word + "_{:06d}".format(int(frame_num) - max_sequence_length)) + ".npy"
                    gt = gt.reshape((max_sequence_length, -1))
                    with open(os.path.join("gt", fname), 'wb') as f:
                        np.save(f, gt)

                    labels[fname] = ["tensor_" + video_name + "_{:06d}".format(int(frame_num) - max_sequence_length) + ".npy",
                                     "mask_" + video_name + "_" + word + "_{:06d}".format(int(frame_num) - max_sequence_length) + ".npy"]

                gt = copy.deepcopy(empty_gt)
                for i in range(max_sequence_length): bbox_temp[i] = [0, 0, 0, 0]
                max_instances_frame = 0

        if (frame_num) % max_sequence_length:
            #print(frame_num, (frame_num - 1) % max_sequence_length, int(frame_num) - int(frame_num) % max_sequence_length + 1)
            if max_instances_frame == 1:
                fname = ("gt_" + video_name + "_" + word + "_{:06d}".format(int(frame_num) - int(frame_num) % max_sequence_length + 1)) + ".npy"
                gt = gt.reshape((max_sequence_length, -1))
                with open(os.path.join("gt", fname), 'wb') as f:
                    np.save(f, gt)

                labels[fname] = ["tensor_" + video_name + "_{:06d}".format(int(frame_num) - int(frame_num) % max_sequence_length + 1) + ".npy",
                                 "mask_" + video_name + "_" + word + "_{:06d}".format(int(frame_num) - int(frame_num) % max_sequence_length + 1) + ".npy"]

            gt = copy.deepcopy(empty_gt)
            for i in range(max_sequence_length): bbox_temp[i] = [0, 0, 0, 0]
            max_instances_frame = 0
print(len(labels.keys()))

with open("gt/labels.json", 'w') as f:
    json.dump(labels, f)







#print("----->", "gt", (video_name + "_gt_" + word + "_{:05d}".format(int(frame_num))) + ".np", frame_num) 


"""fig, ax = plt.subplots(1, figsize=(15,15))
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-max_sequence_length-1);
print(word, frame_num-max_sequence_length-1)

for i in range(max_sequence_length):
    bbox = bbox_temp[i]
    rect = pat.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3] ,linewidth=1,edgecolor='r',facecolor='none')

    ret, inp = cap.read()
    if not ret:
        continue

    im = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    plt.cla()
    ax.imshow(im)

    resized = cv2.resize(gt[i, :, :], (int(video_width), int(video_height)), interpolation = cv2.INTER_AREA) 
    resized.swapaxes(0, 1)
    ax.imshow(resized, cmap='jet', alpha=0.5)
    ax.add_patch(rect)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.001)
plt.close()"""