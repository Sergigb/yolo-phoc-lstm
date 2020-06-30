import cv2
from sklearn.metrics import pairwise_distances
from utils import *
import tensorflow as tf
from yolo_models import build_yolo_v2
from scipy.spatial.distance import cosine
import numpy as np
import glob
import sys
import os

build_phoc = import_cphoc()
img_shape    = (608, 608, 3)
num_priors   = 13
priors       = np.array([(0.67, 0.35), (1.0, 0.52), (1.2, 1.0), (1.34, 0.33), (1.6, 0.58), (2.5, 0.45), (2.24, 0.8), (3.7, 0.79), (3.0, 1.37), (6.0, 1.4), (4.75, 3.0), (10.3, 2.3), (12.0, 5.0)])

phoc_size    = 604
weights_path = './ckpt/yolo-phoc_175800.ckpt'
thresh       = 0.002
n_neighbors  = 10000

model_input  = tf.placeholder(tf.float32, shape=(None,)+img_shape)
model_output = build_yolo_v2(model_input, num_priors, phoc_size)

#import tensorflow.contrib.slim as slim
#model_vars = tf.trainable_variables()
#slim.model_analyzer.analyze_vars(model_vars, print_info=True)


init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

is_test = True

if not is_test:
    tensors_path = "../tensors/train"
    files = glob.glob("../datasets/rrc-text-videos/ch3_train/*.mp4")
else:
    tensors_path = "../tensors/test"
    files = glob.glob("../datasets/rrc-text-videos/ch3_test/*.mp4")

max_sequence_length = 50


with tf.Session() as sess:
    print_info('Loading weights...')
    saver.restore(sess, weights_path)
    print_ok('Done!\n')

    for file_ in files:
        video_name = file_.split("/")[-1].replace(".mp4", "")

        cap = cv2.VideoCapture(file_)
        for i in range(1):
            ret, inp = cap.read()

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tensors_all = []
        descriptors_all = []
        while(ret):
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            inp_feed = np.expand_dims(img_preprocess(cv2.cvtColor(inp,cv2.COLOR_BGR2RGB), shape=img_shape, letterbox=True), 0)
            feed_dict = {model_input : inp_feed}
            out = sess.run(model_output, feed_dict)
            descriptors = out.reshape((-1,phoc_size+5))
            descriptors = descriptors[: , 0:5]
            descriptors = descriptors.flatten()
            descriptors_all.append(descriptors)

            # bn = tf.get_default_graph().get_tensor_by_name('YOLOv2/BatchNorm_21/gamma:0')
            leaky_20 = tf.get_default_graph().get_tensor_by_name('YOLOv2/leaky_20:0')
            tensor = sess.run(leaky_20, feed_dict)
            tensors_all.append(tensor.squeeze(0))

            sys.stdout.write("\r" + str(frame_num) + '/' + str(num_frames))
            sys.stdout.flush()

            ret, inp = cap.read()
        print("")
        tensors_all = np.array(tensors_all)
        descriptors_all = np.array(descriptors_all)

        for index in range(int(num_frames/max_sequence_length)):
            tensor_sequence = tensors_all[max_sequence_length*index:max_sequence_length*(index+1), :]
            descriptors_sequence = descriptors_all[max_sequence_length*index:max_sequence_length*(index+1), :]

            filename = 'tensor_' + video_name + '_' + str((max_sequence_length * index) + 1).zfill(6) + '.npy'
            np.save(os.path.join(tensors_path, filename), tensor_sequence)

            filename = 'descriptors_' + video_name + '_' + str((max_sequence_length * index) + 1).zfill(6) + '.npy'
            np.save(os.path.join(tensors_path, filename), descriptors_sequence)

        if num_frames % max_sequence_length:
            tensor_sequence = tensors_all[max_sequence_length * int(num_frames/max_sequence_length):, :]
            pad = np.zeros((max_sequence_length - tensor_sequence.shape[0], *tensor_sequence.shape[1:]))
            tensor_sequence = np.concatenate((tensor_sequence, pad), axis=0)

            filename = 'tensor_' + video_name + '_' + str((max_sequence_length * int(num_frames/max_sequence_length)) + 1).zfill(6) + '.npy'
            np.save(os.path.join(tensors_path, filename), tensor_sequence)

            filename = 'descriptors_' + video_name + '_' + str((max_sequence_length * int(num_frames/max_sequence_length)) + 1).zfill(6) + '.npy'
            np.save(os.path.join(tensors_path, filename), descriptors_sequence)
