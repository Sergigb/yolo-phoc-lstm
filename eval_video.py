import os
import glob

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

from nms import nms
from random import shuffle

from utils import Sampler, load_descriptors, trans, assign_ids

# descriptors_path = 'extracted_descriptors/extracted_descriptors_361_dist'
descriptors_path = 'extracted_descriptors/extracted_descriptors_361_test'
weights_path = 'models/models_361/model-epoch-40.pth'
num_descriptors = 361

# video_paths = glob.glob('datasets/rrc-text-videos/ch3_train/*.mp4')
video_paths = glob.glob('datasets/rrc-text-videos/ch3_test/*.mp4')
shuffle(video_paths)
# video_paths = ['datasets/rrc-text-videos/ch3_train/Video_10_1_1.mp4']

for video_path in video_paths:
    print(video_path)
    if not os.path.isdir('images'):
        os.mkdir('images')
    files = glob.glob('images/*')
    for f in files: os.remove(f)

    video_name = video_path.split('/')[-1].replace('.mp4', '')
    # voc_path = 'datasets/rrc-text-videos/ch3_train/' + video_name + '_GT.txt'
    voc_path = 'datasets/rrc-text-videos/ch3_test/' + video_name + '_GT_voc.txt'

    cap = cv2.VideoCapture(video_path)
    sampler = Sampler(weights_path=weights_path, num_descriptors=num_descriptors, hidden_size=256, input_size=6)
    tracked_detections = [[] for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]
    _, inp = cap.read()

    queries = set()
    with open(voc_path) as f:
        lines = f.readlines()
    for line in lines:
        word = line.split(',')[-1]
        word = word.translate(trans).lower()
        queries.add(word)

    for word in queries:
        print('query: ' + word)

        # get the predictions from the rnn
        predictions = sampler.sample_video(word, video_name, descriptors_path=descriptors_path)
        # load descriptors
        try:
            descriptors = load_descriptors(video_name, word, descriptors_path, num_descriptors=num_descriptors)
        except:
            print("missing word: " + word)
            continue

        frame = 0
        detections_word = [[] for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            rectongles = []
            scores = []
            for i in range(descriptors.shape[1]):
                if predictions[0, frame+1, i] > 0.25:
                    center_x, center_y, w, h, _, _ = descriptors[frame+1, i]
                    center_x *= inp.shape[1]
                    w *= inp.shape[1]
                    center_y *= inp.shape[0]
                    h *= inp.shape[0]
                    x = center_x - w / 2.
                    y = center_y - h / 2.
                    scores.append(predictions[0, frame + 1, i])
                    rectongles.append(np.array([x, y, x+w ,y+h]))  # sort requires x1, y1, x2, y2, score=1???

            indices = nms.boxes(rectongles, scores)  # non maximal suppression
            for index in indices:
                detections_word[frame].append(rectongles[index])
            frame += 1

        tracked_detections_word = assign_ids(detections_word)  # there might be id collisions
        for i, tracked_detections_frame in enumerate(tracked_detections_word):
            for tracked_detection in tracked_detections_frame:
                tracked_detections[i].append([tracked_detection, word])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    fig, ax = plt.subplots(1, figsize=(15,15))
    ret, inp = cap.read()
    frame = 0

    while ret:
        im = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        plt.cla()
        ax.imshow(im)

        for detection in tracked_detections[frame]:
            x1, y1, x2, y2, boxid = detection[0]
            word = detection[1]
            rect = pat.Rectangle((x1, y1), abs(x2-x1), abs(y2-y1), linewidth=1, edgecolor=(1, 0, 0), facecolor='none')
            plt.text(x1, y1, int(boxid), color=(0, 1, 0), fontsize=20)
            plt.text(x1, y2+15, detection[1], color=(1, 0, 0), fontsize=20)
            ax.add_patch(rect)

        plt.axis('off')
        # plt.show(block=False)
        # plt.pause(0.00001)
        plt.savefig('images/file%05d.jpeg' % frame, bbox_inches = 'tight', pad_inches = 0)

        frame += 1
        ret, inp = cap.read()
    plt.close()
    os.system("ffmpeg -framerate 20 -pattern_type glob -i 'images/*.jpeg' -c:v mpeg4 -vb 1M -qscale:v 2 videos/pointer\ top\ 361\ sparse/" + video_name + ".mp4")
    # os.system("ffmpeg -framerate 25 -pattern_type glob -i 'images/*.png' -c:v mpeg4 -vb 1M -qscale:v 2 video.mp4")


