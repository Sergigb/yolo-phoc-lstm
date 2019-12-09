import os
import glob

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from nms import nms

from utils import Sampler, load_descriptors, trans


# video_path = 'datasets/rrc-text-videos/ch3_test/Video_48_6_4.mp4'
video_path = 'datasets/rrc-text-videos/ch3_train/Video_41_2_3.mp4'
voc_path = 'datasets/rrc-text-videos/ch3_train/Video_41_2_3_GT.txt'
# voc_path = 'datasets/rrc-text-videos/ch3_test/Video_48_6_4_GT_voc.txt'
video_name = video_path.split('/')[-1].replace('.mp4', '')
descriptors_path = 'extracted_descriptors/extracted_descriptors_10'
# descriptors_path = 'extracted_descriptors/extracted_descriptors_10_test'
# weights_path = 'models/model-epoch-200.pth'
weights_path = 'models/best/model-epoch-last.pth'

if not os.path.isdir('images'):
    os.mkdir('images')
files = glob.glob('images/*')
for f in files: os.remove(f)

cap = cv2.VideoCapture(video_path)
sampler = Sampler(weights_path=weights_path)
detections = [[] for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]
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
    descriptors = load_descriptors(video_name, word, descriptors_path)

    frame = 0
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        rectongles = []
        scores = []
        for i in range(descriptors.shape[1]):
            if predictions[0, frame+1, i] > 0.5:
                scores.append(predictions[0, frame+1, i])
                center_x, center_y, w, h, _, _ = descriptors[frame+1, i]
                center_x *= inp.shape[1]
                w *= inp.shape[1]
                center_y *= inp.shape[0]
                h *= inp.shape[0]
                x = center_x - w / 2.
                y = center_y - h / 2.
                rectongles.append([x, y, w ,h])
        indices = nms.boxes(rectongles, scores)  # non maximal suppresion

        for index in indices:
            detections[frame].append([rectongles[index], word])
        frame+=1

cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
fig, ax = plt.subplots(1)
ret, inp = cap.read()
frame = 0

while ret:
    im = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    plt.cla()
    ax.imshow(im)

    for detection in detections[frame]:
        x, y, w, h = detection[0]
        rect = pat.Rectangle((x, y), abs(w), abs(h), linewidth=1, edgecolor=(1, 0, 0), facecolor='none')
        plt.text(x, y, detection[1], color=(1, 0, 0), fontsize=10)
        ax.add_patch(rect)

    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.00001)
    # plt.savefig('images/file%05d.png' % frame)

    frame += 1
    ret, inp = cap.read()


