import copy
import os
import json

import xml.etree.cElementTree as ET
import glob
import numpy as np
import cv2

from utils import trans, iou

n_descriptors = 10
labels_fname = 'gt/labels-100-top' + str(n_descriptors) + '.json'
dataset_path = 'datasets/rrc-text-videos/ch3_train/'
descriptor_root = 'extracted_descriptors/extracted_descriptors_' + str(n_descriptors)
gt_path = 'gt/'
annotations_paths = glob.glob(dataset_path + '*.xml')
# annotations_paths = ['datasets/rrc-text-videos/ch3_train/Video_37_2_3_GT.xml']
if not os.path.exists(gt_path):
    os.mkdir(gt_path)

threshold = 0.5
max_sequence_length = 100  # divide the annotations of the videos in sequences
labels = {}
total_objects = 0
empty_mask = np.zeros((max_sequence_length, n_descriptors))

for annotations_path in annotations_paths:
    print('Processing file ' + annotations_path)

    video_path = annotations_path.replace('_GT.xml', '.mp4')
    cap = cv2.VideoCapture(video_path)
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ret, inp = cap.read()

    # load the vocabulary from the gt
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
        has_instances = False  # to keep track of whether the current sequence has this word or not
        current_frame = 0
        mask = copy.deepcopy(empty_mask)

        # load descriptors
        video_name = video_path.split('/')[-1].replace('.mp4', '')
        descriptor_file = descriptor_root + '/descriptors_top' + str(n_descriptors) + '_' + video_name + '_' + word + '*'
        files = glob.glob(descriptor_file)
        files = sorted(files)
        descriptors = None
        for file in files:
            if descriptors is None:
                descriptors = np.load(file).squeeze()
            else:
                descriptors = np.concatenate((descriptors, np.load(file).squeeze()))
        descriptors = descriptors.reshape((descriptors.shape[0], int(descriptors.shape[1]/6), 6))

        for frame in root:
            current_frame = int(frame.get('ID'))
            annotations_frame = []
            # get all the annotated objects in this frame
            for object_ in frame:
                if object_.get('Transcription').translate(trans).lower() == word:
                    has_instances = True
                    x_coords = []
                    y_coords = []
                    for point in object_.iter('Point'):
                        x_coords.append(int(point.get('x')))
                        y_coords.append(int(point.get('y')))
                    x1 = min(x_coords) / video_width
                    y1 = min(y_coords) / video_height
                    x2 = max(x_coords) / video_width
                    y2 = max(y_coords) / video_height
                    bbox = np.array([x1, y1, x2, y2])
                    annotations_frame.append(bbox)

            # compute the iou between the detections and the gt
            mask_frame = np.zeros([descriptors.shape[1]])
            for i in range(descriptors.shape[1]):
                x, y, w, h, _, _ = descriptors[current_frame-1, i]
                x1 = x - w/2
                y1 = y - h/2
                x2 = x1 + w
                y2 = y1 + h
                bbox_pred = [x1, y1, x2, y2]
                for annotation in annotations_frame:
                    iou_score = iou(bbox_pred, annotation[0:4])
                    # we don't care about overlaps, only if the current detection is over a bbox from the gt
                    if iou_score > threshold:
                        mask_frame[i] = 1
            mask[(current_frame % max_sequence_length) - 1, :] = mask_frame

            if not current_frame % max_sequence_length:
                ann_fname = 'mask_top' + str(n_descriptors) + '_' + video_name + '_' + word + '_' + \
                            str(max_sequence_length) + '_' + str((current_frame - max_sequence_length) + 1).zfill(6) + '.npy'
                if has_instances:  # we save all the masks but only include them in the labels if the sequence contains the word
                    labels[ann_fname] = 'descriptors_top' + str(n_descriptors) + '_' + video_name + '_' + word + \
                                        '_' + str((current_frame - max_sequence_length) + 1).zfill(6) + '.npy'
                    total_objects += 1
                np.save(os.path.join(gt_path, ann_fname), mask)

                mask = copy.deepcopy(empty_mask)
                has_instances = False


        if current_frame % max_sequence_length:
            ann_fname = 'mask_top' + str(n_descriptors) + '_' +  video_name + '_' + word + '_' + str(max_sequence_length)\
                        + '_' + str(int(current_frame) - int(current_frame) % max_sequence_length + 1).zfill(6) + '.npy'
            if has_instances:
                labels[ann_fname] = 'descriptors_top' + str(n_descriptors) + '_' + video_name + '_' + word + '_' + \
                                    str(int(current_frame) - int(current_frame) % max_sequence_length + 1).zfill(6) + '.npy'
                total_objects += 1
            np.save(os.path.join(gt_path, ann_fname), mask)

print('Number of valid sequences: ' + str(total_objects))

with open(labels_fname, 'w') as f:
    json.dump(labels, f)
