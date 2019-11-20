import copy
import os
import json

import xml.etree.cElementTree as ET
import glob
import numpy as np
import cv2


labels_fname = 'gt/labels-100-top100.json'
dataset_path = 'datasets/rrc-text-videos/ch3_train/'
gt_path = 'gt/'
annotations_paths = glob.glob(dataset_path + '*.xml')
# annotations_paths = ['../datasets/rrc-text-videos/ch3_train/Video_41_2_3_GT.xml']
if not os.path.exists(gt_path):
    os.mkdir(gt_path)

trans = str.maketrans({'.': r'', '"': r'', '\n': r'', '-': r'', '\'': r''})
max_sequence_length = 100  # divide the annotations of the videos in sequences of 250 frames at max
empty_annotations = np.zeros((max_sequence_length, 5))  # empty annotations array
labels = {}
total_objects = 0

for annotations_path in annotations_paths:
    print('Processing file ' + annotations_path)

    video_path = annotations_path.replace('_GT.xml', '.mp4')
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    vocabulary = set()
    voc_path = annotations_path.replace('.xml', '.txt')
    with open(voc_path) as f:
        lines = f.readlines()
    for line in lines:
        word = line.split(',')[-1]
        word = word.translate(trans).lower()
        vocabulary.add(word)

    tree = ET.parse(annotations_path)
    root = tree.getroot()

    for word in vocabulary:
        previous_object_id = None  # use the annotation with the same id as the one used in the previous frame
        has_instances = False  # to keep track of whether the current sequence has this word or not
        repeated_instance = False  # checks if the current frame has the same word multiple times
        annotations = copy.deepcopy(empty_annotations)
        current_frame = 0
        for frame in root:
            current_frame = frame.get('ID')
            valid_instances = 0
            for object_ in frame:
                if object_.get('Transcription').translate(trans).lower() == word:
                    valid_instances += 1
                    has_instances = True
                    if previous_object_id is None:
                        previous_object_id = object_.get('ID')
                    if valid_instances > 1:
                        repeated_instance = True
                    x_coords = []
                    y_coords = []
                    for point in object_.iter('Point'):
                        x_coords.append(int(point.get('x')))
                        y_coords.append(int(point.get('y')))
                    w = (max(x_coords) - min(x_coords))
                    h = (max(y_coords) - min(y_coords))
                    x = (min(x_coords) + w/2.) / width
                    y = (min(y_coords) + h/2.) / height
                    w = w / width
                    h = h / height
                    bbox = np.array([x, y, w, h, 1])  # [center x, center y, w, h, objectness]
                    annotations[(int(current_frame) % max_sequence_length) - 1] = bbox
                    if previous_object_id == object_.get('ID'):
                        break  # ignore the rest of the bboxes of the same word in the current frame
            if has_instances and not int(current_frame) % max_sequence_length and not repeated_instance:
                video_name = annotations_path.split('/')[-1].replace('_GT.xml', '')
                ann_fname = 'GT_' + video_name + '_' + word + '_' + str(max_sequence_length) + '_' + \
                            str(int(current_frame) - max_sequence_length + 1) + '.npy'
                labels[ann_fname] = 'descriptors_top100_' + video_name + '_' + word + '_' + \
                                    str(int(current_frame) - max_sequence_length + 1) + '.npy'

                np.save(os.path.join(gt_path, ann_fname), annotations)
                annotations = copy.deepcopy(empty_annotations)

                has_instances = False
                previous_object_id = None
                repeated_instance = False
                total_objects += 1

        if has_instances and int(current_frame) % max_sequence_length and not repeated_instance:
            video_name = annotations_path.split('/')[-1].replace('_GT.xml', '')
            ann_fname = 'GT_' + video_name + '_' + word + '_' + str(max_sequence_length) + '_' + \
                        str(int(current_frame) - int(current_frame) % max_sequence_length + 1) + '.npy'
            labels[ann_fname] = 'descriptors_top100_' + video_name + '_' + word + '_' + \
                                str(int(current_frame) - int(current_frame) % max_sequence_length + 1) + '.npy'
            np.save(os.path.join(gt_path, ann_fname), annotations)

            total_objects += 1
print('Number of valid sequences: ' + str(total_objects))

with open(labels_fname, 'w') as f:
    json.dump(labels, f)
