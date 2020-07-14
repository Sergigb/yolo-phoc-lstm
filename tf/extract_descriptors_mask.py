import cv2
from utils import *
import tensorflow as tf
from yolo_models import build_yolo_v2
import numpy as np
import glob
import os
from sklearn.metrics import pairwise_distances
import sys



def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    bbox1area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou_score = intersection / float((bbox1area + bbox2area) - intersection)
    return iou_score


build_phoc = import_cphoc()
img_shape    = (608, 608, 3)
num_priors   = 13
priors       = np.array([(0.67, 0.35), (1.0, 0.52), (1.2, 1.0), (1.34, 0.33), (1.6, 0.58), (2.5, 0.45), (2.24, 0.8),
                         (3.7, 0.79), (3.0, 1.37), (6.0, 1.4), (4.75, 3.0), (10.3, 2.3), (12.0, 5.0)])
phoc_size    = 604
max_sequence_length = 50
att_size = 38

weights_path = './ckpt/yolo-phoc_175800.ckpt'
model_input = tf.placeholder(tf.float32, shape=(None,)+img_shape)
model_output = build_yolo_v2(model_input, num_priors, phoc_size)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

trans = str.maketrans({'.': r'', '"': r'', '\n': r'', '-': r'', '\'': r''})

is_test = False

if is_test:
    masks_path = '../tensors/test'
    video_files_path = '../datasets/rrc-text-videos/ch3_test/'  # test
else:
    masks_path = '../tensors/train'
    video_files_path = '../datasets/rrc-text-videos/ch3_train/'  # train

video_paths = glob.glob(video_files_path + '*.mp4')
#video_paths = ['../datasets/rrc-text-videos/ch3_train/Video_45_6_4.mp4']

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

step = 1 / att_size

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    print_info('Loading weights...')
    saver.restore(sess, weights_path)
    print_ok('Done!\n')
    for video_path in video_paths:
        print('Processing file ' + video_path)
        words = set()
        masks = dict()

        gt_path = video_path.replace('.mp4', '_GT_voc.txt')
        with open(gt_path) as f:
            lines = f.readlines()
        for line in lines:
            word = str.encode(line.translate(trans).lower())
            words.add(word)
            masks[word] = []
        #words = [b"caprabo"]

        cap = cv2.VideoCapture(video_path)
        ret, inp = cap.read()
        frame = 1

        while ret:
            inp_feed = np.expand_dims(img_preprocess(cv2.cvtColor(inp,cv2.COLOR_BGR2RGB),
                                                     shape=img_shape, letterbox=True), 0)
            feed_dict = {model_input: inp_feed}
            out = sess.run(model_output, feed_dict)
            descriptors_frame = []

            w = float(out.shape[2])
            h = float(out.shape[1])
            img_w = inp.shape[1]
            img_h = inp.shape[0]
            ar = img_w / img_h  # used recover the original y component and height of the bboxes
            half_padd = ((img_w - img_h) / 2) / img_h

            for i in range(out.shape[1]):
                for j in range(out.shape[2]):
                    max_o = 0.0
                    max_o_descriptor = None

                    col = float(j)
                    row = float(i)
                    for a in range(num_priors):
                        index = a * (phoc_size + 5)
                        out[0, i, j, index:index + 2] = expit(out[0, i, j, index:index + 2])
                        # phoc expit, I guess it went thru logit to avoid negative sqrt?:
                        out[0, i, j, index + 4:index + phoc_size + 5] = \
                            expit(out[0, i, j, index + 4:index + phoc_size + 5])

                        if out[0, i, j, index+4] <  max_o:
                            continue
                        max_o = out[0, i, j, index+4]

                        if img_h > img_w: print("height is bigger than width")

                        # normalized descriptors
                        out[0, i, j, index] = ((col + out[0, i, j, index]) / w)
                        out[0, i, j, index + 1] = ((row + out[0, i, j, index + 1]) / h)  # y comp. padded, wrong a/r:
                        out[0, i, j, index + 1] = (out[0, i, j, index + 1] * ar) - half_padd
                        out[0, i, j, index + 2] = (np.exp(out[0, i, j, index + 2]) * priors[a, 0] / w)
                        out[0, i, j, index + 3] = (np.exp(out[0, i, j, index + 3]) * priors[a, 1] / h) * ar

                        max_o_descriptor = out[0, i, j, index:index + phoc_size + 5]
                    descriptors_frame.append(max_o_descriptor)

            descriptors_frame = np.array(descriptors_frame)

            for word in words:
                q = np.array(build_phoc(word)).reshape(1, -1)
                distances = pairwise_distances(descriptors_frame[:, 5:], q, metric='cosine')
                descriptor = np.concatenate((descriptors_frame[:, :5], distances), axis=1)  # not needed, we can work on distances and descriptors without concat
                mask = np.zeros((att_size, att_size))

                valid_desc = descriptor[(descriptor[:, 4] > 0.5) & (descriptor[:, 5] < 0.5)]

                # this is disgusting
                if(len(valid_desc)):
                    for k in range(valid_desc.shape[0]):
                        temp_mask = np.zeros((att_size, att_size))
                        bbox = [valid_desc[k, 0] - (valid_desc[k, 2] / 2), valid_desc[k, 1] - (valid_desc[k, 2] / 2),\
                                valid_desc[k, 0] + (valid_desc[k, 2] / 2), valid_desc[k, 1] + (valid_desc[k, 2] / 2)]

                        for i in range(0, att_size):
                            for j in range(0, att_size):
                                x = i * step
                                y = j * step
                                if iou([x, y, x+step, y+step], bbox):
                                    temp_mask[j, i] = valid_desc[k, 4] # objectness??? why not
                        mask = np.maximum(mask, temp_mask)

                masks[word].append(mask)

            sys.stdout.write('\rProgress: ' + str(frame) + '/' + str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
            sys.stdout.flush()
            ret, inp = cap.read()
            frame += 1
        print('')

        video_fname = video_path.split('/')[-1].replace('.mp4', '')
        for key in masks.keys():
            masks_key = np.array([masks[key]])
            num_frames = masks_key.shape[1]
            for index in range(int(num_frames/max_sequence_length)):
                masks_sequence = masks_key[:, max_sequence_length*index:max_sequence_length*(index+1)]
                shape = masks_sequence.shape

                filename = 'mask_' + video_fname + '_' + key.decode('utf-8') \
                           + '_' + str((max_sequence_length * index) + 1).zfill(6) + '.npy'
                np.save(os.path.join(masks_path, filename), masks_sequence)

            if num_frames % max_sequence_length:
                masks_sequence = masks_key[:, max_sequence_length * int(num_frames/max_sequence_length):]
                shape = masks_sequence.shape
                pad = np.zeros((1, max_sequence_length - shape[1], shape[2], shape[3]))
                masks_sequence = np.concatenate((masks_sequence, pad), axis=1)

                filename = 'mask_' + video_fname + '_' + key.decode('utf-8') \
                           + '_' + str((max_sequence_length * int(num_frames/max_sequence_length)) + 1).zfill(6) + '.npy'
                np.save(os.path.join(masks_path, filename), np.array(masks_sequence))

