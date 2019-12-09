import cv2
from utils import *
import tensorflow as tf
from yolo_models import build_yolo_v2
import numpy as np
import glob
import os
from sklearn.metrics import pairwise_distances
import sys


build_phoc = import_cphoc()
img_shape    = (608, 608, 3)
num_priors   = 13
priors       = np.array([(0.67, 0.35), (1.0, 0.52), (1.2, 1.0), (1.34, 0.33), (1.6, 0.58), (2.5, 0.45), (2.24, 0.8),
                         (3.7, 0.79), (3.0, 1.37), (6.0, 1.4), (4.75, 3.0), (10.3, 2.3), (12.0, 5.0)])
phoc_size    = 604
max_sequence_length = 100
n_descriptors = 25

weights_path = './ckpt/yolo-phoc_175800.ckpt'
model_input = tf.placeholder(tf.float32, shape=(None,)+img_shape)
model_output = build_yolo_v2(model_input, num_priors, phoc_size)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

trans = str.maketrans({'.': r'', '"': r'', '\n': r'', '-': r'', '\'': r''})

is_test = False

if is_test:
    descriptors_path = '../extracted_descriptors/extracted_descriptors_' + str(n_descriptors) + '_test'
    video_files_path = '../datasets/rrc-text-videos/ch3_test/'  # test
else:
    descriptors_path = '../extracted_descriptors/extracted_descriptors_' + str(n_descriptors)
    video_files_path = '../datasets/rrc-text-videos/ch3_train/'  # train

video_paths = glob.glob(video_files_path + '*.mp4')
# video_paths = ['../datasets/rrc-text-videos/ch3_train/Video_8_5_1.mp4']

if not os.path.isdir(descriptors_path):
    os.mkdir(descriptors_path)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    print_info('Loading weights...')
    saver.restore(sess, weights_path)
    print_ok('Done!\n')
    for video_path in video_paths:
        print('Processing file ' + video_path)
        words = set()
        descriptors = dict()

        if not is_test:
            gt_path = video_path.replace('.mp4', '_GT.txt')
            with open(gt_path) as f:
                lines = f.readlines()
            for line in lines:
                word = line.split(',')[-1]
                word = str.encode(word.translate(trans).lower())
                words.add(word)
                descriptors[word] = []
        else:
            gt_path = video_path.replace('.mp4', '_GT_voc.txt')
            with open(gt_path) as f:
                lines = f.readlines()
            for line in lines:
                word = str.encode(line.translate(trans).lower())
                words.add(word)
                descriptors[word] = []

        cap = cv2.VideoCapture(video_path)
        ret, inp = cap.read()
        frame = 1

        while ret:
            inp_feed = np.expand_dims(img_preprocess(cv2.cvtColor(inp,cv2.COLOR_BGR2RGB),
                                                     shape=img_shape, letterbox=True), 0)
            feed_dict = {model_input: inp_feed}
            out = sess.run(model_output, feed_dict)
            for i in range(out.shape[1]):
                for j in range(out.shape[2]):
                    for a in range(num_priors):
                        index = a * (phoc_size + 5)
                        out[0, i, j, index:index + 2] = expit(out[0, i, j, index:index + 2])
                        # phoc expit, I guess it went thru logit to avoid negative sqrt?:
                        out[0, i, j, index + 4:index + phoc_size + 5] = \
                            expit(out[0, i, j, index + 4:index + phoc_size + 5])
                        col = float(j)
                        row = float(i)
                        w = float(out.shape[2])
                        h = float(out.shape[1])
                        img_w = inp.shape[1]
                        img_h = inp.shape[0]

                        ar = img_w / img_h  # used recover the original y component and height of the bboxes
                        half_padd = ((img_w - img_h) / 2) / img_h

                        if img_h > img_w: print("height is bigger than width")

                        # normalized descriptors
                        out[0, i, j, index + 0] = ((col + out[0, i, j, index + 0]) / w)
                        out[0, i, j, index + 1] = ((row + out[0, i, j, index + 1]) / h)  # y comp. padded, wrong a/r:
                        out[0, i, j, index + 1] = (out[0, i, j, index + 1] * ar) - half_padd
                        out[0, i, j, index + 2] = (np.exp(out[0, i, j, index + 2]) * priors[a, 0] / w)
                        out[0, i, j, index + 3] = (np.exp(out[0, i, j, index + 3]) * priors[a, 1] / h) * ar

            out = out.reshape((-1, phoc_size + 5))
            out = out[out[:, 4].argsort()[::-1]]  # keep the top n descriptors sorted by the objectness
            out = out[0:n_descriptors, :]

            for word in words:
                q = np.array(build_phoc(word)).reshape(1, -1)
                distances = pairwise_distances(out[:, 5:], q, metric='cosine')
                descriptor = np.concatenate((out[:, :5], distances), axis=1)
                descriptors[word].append(descriptor)

            sys.stdout.write('\rProgress: ' + str(frame) + '/' + str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
            sys.stdout.flush()
            ret, inp = cap.read()
            frame += 1
        print('')

        video_fname = video_path.split('/')[-1].replace('.mp4', '')
        for key in descriptors.keys():
            descriptors_key = np.array([descriptors[key]])
            num_frames = descriptors_key.shape[1]
            for index in range(int(num_frames/max_sequence_length)):
                descriptors_sequence = descriptors_key[:, max_sequence_length*index:max_sequence_length*(index+1)]
                shape = descriptors_sequence.shape
                descriptors_sequence = descriptors_sequence.reshape(shape[0], shape[1], shape[2]*shape[3])

                filename = 'descriptors_top' + str(n_descriptors) + '_' + video_fname + '_' + key.decode('utf-8') \
                           + '_' + str((max_sequence_length * index) + 1).zfill(6) + '.npy'
                np.save(os.path.join(descriptors_path, filename), descriptors_sequence)

            if num_frames % max_sequence_length:
                descriptors_sequence = descriptors_key[:, max_sequence_length * int(num_frames/max_sequence_length):]
                shape = descriptors_sequence.shape
                descriptors_sequence = descriptors_sequence.reshape(shape[0], shape[1], shape[2]*shape[3])
                pad = np.zeros((1, max_sequence_length - shape[1], shape[2]*shape[3]))
                descriptors_sequence = np.concatenate((descriptors_sequence, pad), axis=1)

                filename = 'descriptors_top' + str(n_descriptors) + '_'  + video_fname + '_' + key.decode('utf-8') \
                           + '_' + str((max_sequence_length * int(num_frames/max_sequence_length)) + 1).zfill(6) + '.npy'
                np.save(os.path.join(descriptors_path, filename), np.array(descriptors_sequence))

