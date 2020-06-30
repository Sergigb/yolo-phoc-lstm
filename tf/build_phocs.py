import cv2
from utils import *
import numpy as np
import glob
import os
import sys

trans = str.maketrans({'.': r'', '"': r'', '\n': r'', '-': r'', '\'': r''})

build_phoc = import_cphoc()

is_test = False

if not is_test:
    video_paths = glob.glob("../datasets/rrc-text-videos/ch3_train/*.mp4")
else:
    video_paths = glob.glob("../datasets/rrc-text-videos/ch3_test/*.mp4")



for video_path in video_paths:
    print('Processing file ' + video_path)
    words = set()

    gt_path = video_path.replace('.mp4', '_GT_voc.txt')
    with open(gt_path) as f:
        lines = f.readlines()
    for line in lines:
        word = str.encode(line.translate(trans).lower())
        words.add(word)

    for word in words:
        q = np.array(build_phoc(word)).reshape(1, -1)
        np.save("../phocs/" + word.decode('utf-8') + ".npy", q)




