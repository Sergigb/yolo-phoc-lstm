import os
import glob

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat

from nms import nms

from utils import Sampler, load_descriptors, trans, assign_ids

# video_path = 'datasets/rrc-text-videos/ch3_test/Video_43_6_4.mp4'
# video_path = 'datasets/rrc-text-videos/ch3_train/Video_42_2_3.mp4'
# voc_path = 'datasets/rrc-text-videos/ch3_test/Video_43_6_4_GT_voc.txt'
descriptors_path = 'extracted_descriptors/extracted_descriptors_361'
# descriptors_path = 'extracted_descriptors/extracted_descriptors_10_test'
# weights_path = 'models/model-epoch-200.pth'
# weights_path = 'models/cluster/model-epoch-last.pth'
weights_path = 'models/model-epoch-105.pth'
num_descriptors = 361

# video_paths = glob.glob('datasets/rrc-text-videos/ch3_train/*.mp4')
video_paths = ['datasets/rrc-text-videos/ch3_train/Video_37_2_3.mp4']

for video_path in video_paths:
    print(video_path)
    if not os.path.isdir('images'):
        os.mkdir('images')
    files = glob.glob('images/*')
    for f in files: os.remove(f)

    video_name = video_path.split('/')[-1].replace('.mp4', '')
    voc_path = 'datasets/rrc-text-videos/ch3_train/' + video_name + '_GT.txt'

    cap = cv2.VideoCapture(video_path)
    sampler = Sampler(weights_path=weights_path, num_descriptors=num_descriptors, hidden_size=256, input_size=16)
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
        descriptors = load_descriptors(video_name, word, descriptors_path, num_descriptors=num_descriptors)

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
                    rectongles.append([x, y, x+w ,y+h])  # sort requires x1, y1, x2, y2
            indices = nms.boxes(rectongles, scores)  # non maximal suppression

            for index in indices:
                detections[frame].append([rectongles[index], word])  # TODO: APPEND SCORE
            frame+=1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    fig, ax = plt.subplots(1, figsize=(10,10))
    ret, inp = cap.read()
    frame = 0

    tracked_detections = assign_ids(detections)  # I should do this for every word like with nms

    while ret:
        im = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        plt.cla()
        ax.imshow(im)

        # for detection, untracked_det in zip(tracked_detections[frame], detections[frame]):
        for detection in detections[frame]:
            # x1, y1, x2, y2, id = detection
            x1, y1, x2, y2, _ = detection[0]
            id = detection[1]
            rect = pat.Rectangle((x1, y1), abs(x2-x1), abs(y2-y1), linewidth=1, edgecolor=(1, 0, 0), facecolor='none')
            plt.text(x1, y1, str(id), color=(0, 1, 0), fontsize=10)
            # plt.text(x1, y2+20, untracked_det[1], color=(1, 0, 0), fontsize=10)
            ax.add_patch(rect)

        plt.axis('off')
        # plt.show(block=False)
        # plt.pause(0.00001)
        plt.savefig('images/file%05d.jpeg' % frame)

        frame += 1
        ret, inp = cap.read()
    plt.close()
    os.system("ffmpeg -framerate 20 -pattern_type glob -i 'images/*.jpeg' -c:v mpeg4 -vb 1M -qscale:v 2 videos/pointer\ top\ 361/" + video_name + ".mp4")
    # os.system("ffmpeg -framerate 25 -pattern_type glob -i 'images/*.png' -c:v mpeg4 -vb 1M -qscale:v 2 video.mp4")


