import cv2
from sklearn.metrics import pairwise_distances
from utils import *
import tensorflow as tf
from yolo_models import build_yolo_v2
from scipy.spatial.distance import cosine
import time
import numpy as np

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

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

for query in [b'nuss']:
    q = np.array(build_phoc(query.lower())).reshape(1,-1)
    with tf.Session() as sess:
        print_info('Loading weights...')
        saver.restore(sess, weights_path)
        print_ok('Done!\n')
        cap = cv2.VideoCapture('../datasets/rrc-text-videos/ch3_train/Video_46_6_4.mp4')
        for i in range(1):
            ret, inp = cap.read()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vout = cv2.VideoWriter('nuss.avi', fourcc, 24.0, (608,608))
        while(ret):
            ini=time.time()
            inp_feed = np.expand_dims(img_preprocess(cv2.cvtColor(inp,cv2.COLOR_BGR2RGB), shape=img_shape, letterbox=True), 0)
            feed_dict = {model_input : inp_feed}
            out = sess.run(model_output, feed_dict)
            descriptors = out.reshape((-1,phoc_size+5))
            descriptors = expit(descriptors)
            valid_descriptors = descriptors[tuple(np.where(descriptors[:,4] > thresh)[0]), 5:]
            d=pairwise_distances(valid_descriptors,q,metric='cosine')

            bboxes = []
            phocs  = []
            for i in range(out.shape[1]):
                for j in range(out.shape[2]):
                    for a in range(num_priors):
                        index = a*(phoc_size+5)
                        out[0,i,j,index:index+2] = expit(out[0,i,j,index:index+2])
                        out[0,i,j,index+4:index+phoc_size+5] = expit(out[0,i,j,index+4:index+phoc_size+5])
                        col = float(j)
                        row = float(i)
                        w   = float(out.shape[2])
                        h   = float(out.shape[1])
                        img_w = img_shape[1]
                        img_h = img_shape[0]

                        out[0,i,j,index+0] = ((col + out[0,i,j,index+0]) / w) * img_w
                        out[0,i,j,index+1] = ((row + out[0,i,j,index+1]) / h) * img_h
                        out[0,i,j,index+2] = (np.exp(out[0,i,j,index+2]) * priors[a,0] / w) * img_w
                        out[0,i,j,index+3] = (np.exp(out[0,i,j,index+3]) * priors[a,1] / h) * img_h

                        left  = (out[0,i,j,index+0]-out[0,i,j,index+2]/2.)
                        right = (out[0,i,j,index+0]+out[0,i,j,index+2]/2.)
                        top   = (out[0,i,j,index+1]-out[0,i,j,index+3]/2.)
                        bot   = (out[0,i,j,index+1]+out[0,i,j,index+3]/2.)
                        if out[0,i,j,index+4] > thresh:
                            bboxes.append((left,right,top,bot))
                            phocs.append(out[0,i,j,index+5:index+phoc_size+5])

            im = inp.copy()
            top_border, bottom_border, left_border, right_border = (0,0,0,0)
            h,w,c = im.shape
            letterbox = cv2.resize(im, (img_shape[1], int(float(h)*(float(img_shape[1])/w))))
            top_border    = int( float(img_shape[0] - int(float(h)*(float(img_shape[1])/w))) / 2. )
            bottom_border = top_border
            im = cv2.copyMakeBorder(letterbox, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT, value=[127,127,127])

            if np.min(d)<.35:
                heatmap = np.zeros(im.shape)
                for i,b in enumerate(bboxes):
                    alpha = 1-cosine(q,phocs[i])
                    heatmap[int(b[2]):int(b[3]), int(b[0]):int(b[1])] += alpha
                heatmap = (heatmap / np.max(heatmap))
                heatmap = np.array((heatmap*255), dtype=np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                im = cv2.addWeighted(heatmap, 0.5, im, 0.5, 0)
                cv2.circle(im, (30,30), 20, (0,255,0), -1)
                cv2.putText(im,' '+str(query), (50,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

            else:
                cv2.circle(im, (30,30), 20, (0,0,255), -1)
                cv2.putText(im,' '+str(query), (50,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            # cv2.imshow('Display',im)
            print(im.shape)
            vout.write(im)
            # cv2.waitKey(10)
            ret, inp = cap.read()
            end = time.time()
            print('Elapsed '+str(end-ini)+' secs.')

    vout.release()
