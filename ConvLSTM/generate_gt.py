import random
import copy
import os
import json
from math import ceil, floor

import xml.etree.cElementTree as ET
import glob
import numpy as np
import cv2
from matplotlib import patches as pat, pyplot as plt


dataset_path = "../datasets/rrc-text-videos/ch3_train/"
annotations_paths = glob.glob(dataset_path + "*.xml")
max_seq_length = 100

annotations = {} # id, dict with stats

for annotations_path in annotations_paths:
    print("Processing file " + annotations_path)

    video_path = annotations_path.replace("_GT.xml", ".mp4")
    cap = cv2.VideoCapture(video_path)
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    tree = ET.parse(annotations_path)
    root = tree.getroot()

    annotations_frame = {}

    vocabulary = {}
    voc_path = annotations_path.replace('.xml', '.txt')
    with open(voc_path) as f:
        lines = f.readlines()
    for line in lines:
        voc_id = line.split(',')[0].replace("\"", "").replace("\n", "")
        word = line.split(',')[-1].replace("\"", "").replace("\n", "")
        vocabulary[voc_id] = word

    for frame in root:
        current_frame = int(frame.get("ID")) - 1
        for object_ in frame:
            if object_.get("ID") not in vocabulary.keys():
                continue
            if object_.get("Quality") in ("MODERATE", "HIGH") and vocabulary[object_.get("ID")] == object_.get("Transcription"):
                object_id = object_.get("ID")

                x_coords = []
                y_coords = []
                for point in object_.iter('Point'):
                    x_coords.append(int(point.get('x')))
                    y_coords.append(int(point.get('y')))

                x1 = max(min(min(x_coords) / video_width, 1), 0)
                y1 = max(min(min(y_coords) / video_height, 1), 0)
                x2 = max(min(max(x_coords) / video_width, 1), 0)
                y2 = max(min(max(y_coords) / video_height, 1), 0)

                bbox = np.array([x1, y1, x2, y2])

                if object_id not in annotations_frame.keys():
                    stats = {"transcription": object_.get("Transcription"),
                             "first": current_frame,
                             "last": current_frame,
                             "bboxes": {}}
                    annotations_frame[object_id] = stats
                    annotations_frame[object_id]["bboxes"][current_frame] = bbox
                else:
                    annotations_frame[object_id]["last"] = current_frame
                    annotations_frame[object_id]["bboxes"][current_frame] = bbox


    annotations[video_path.split("/")[-1].replace(".mp4", "")] = annotations_frame

training_samples = []
gt_dict = {}

for video_name in annotations.keys():
    for object_id in annotations[video_name]:
        annotation = annotations[video_name][object_id]

        seq_start = annotation["first"]
        seq_end = annotation["last"]
        gt = np.zeros((max_seq_length, 38, 38))

        if(seq_end - seq_start) == 0: continue  # only 1 frame, discard

        if seq_end > seq_start + max_seq_length:
            seq_end = seq_start + max_seq_length  # discard the rest, not ideal but whatever
        else:
            seq_end = seq_end + 1

        sample = {"video_name": video_name,
                  "object_id": object_id,
                  "seq_start": seq_start,
                  "seq_end": seq_end,
                  "transcription": annotation["transcription"]}

        for i in range(seq_end - seq_start):
            if i + seq_start in annotation["bboxes"]:
                x1, y1, x2, y2 = annotation["bboxes"][i + seq_start] * 38
                gt[i, floor(y1):ceil(y2), floor(x1):ceil(x2)] = 1

        gt_dict[video_name + "_" + object_id] = gt
        training_samples.append(sample)
print(len(training_samples))

if not os.path.exists("gt"):
    os.mkdir("gt")

for key in gt_dict.keys():
    np.save("gt/" + key + ".npy", gt_dict[key])

with open("gt/gt.json", "w") as f:
    json.dump(training_samples, f)

# visualize
while(0):
    sample = random.choice(training_samples)
    video_name = sample["video_name"]
    object_id = sample["object_id"]
    video_path = dataset_path + video_name + ".mp4"

    cap = cv2.VideoCapture(video_path)
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cap.set(cv2.CAP_PROP_POS_FRAMES, sample["seq_start"])

    fig, ax = plt.subplots(1)
    ret, inp = cap.read()
    current_frame = sample["seq_start"]
    gt = gt_dict[video_name + "_" + object_id]

    for i in range(sample["seq_end"] - sample["seq_start"]):
        im = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        plt.cla()
        ax.imshow(im)

        gt_resized = cv2.resize(gt[i, :, :], (int(video_width), int(video_height)), interpolation = cv2.INTER_AREA)
        ax.imshow(gt_resized, cmap='jet', alpha=0.5)

        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.00001)

        print(sample["transcription"], sample["seq_end"] - sample["seq_start"])
        ret, inp = cap.read()
        current_frame += 1

    plt.close()
    cap.release()



"""
Video_42_2_3.mp4 1005 {'Transcription': 'mundi', 'First': 0, 'Last': 118, 'Howmany': 119}
Video_42_2_3.mp4 1007 {'Transcription': 'DDP', 'First': 0, 'Last': 189, 'Howmany': 190}
Video_42_2_3.mp4 1008 {'Transcription': 'LOOK', 'First': 0, 'Last': 9, 'Howmany': 10}
Video_42_2_3.mp4 1009 {'Transcription': 'VOYAGES', 'First': 0, 'Last': 9, 'Howmany': 10}
Video_42_2_3.mp4 4002 {'Transcription': 'Les', 'First': 3, 'Last': 243, 'Howmany': 239}
Video_42_2_3.mp4 4003 {'Transcription': 'createurs', 'First': 3, 'Last': 242, 'Howmany': 240}
Video_42_2_3.mp4 4004 {'Transcription': 'de', 'First': 3, 'Last': 246, 'Howmany': 244}
Video_42_2_3.mp4 4005 {'Transcription': 'Marie', 'First': 3, 'Last': 101, 'Howmany': 99}
Video_42_2_3.mp4 33006 {'Transcription': 'mundi', 'First': 32, 'Last': 111, 'Howmany': 80}
Video_42_2_3.mp4 33007 {'Transcription': 'harmonia', 'First': 32, 'Last': 100, 'Howmany': 69}
"""