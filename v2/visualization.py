import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# gt tensors
#gt = np.concatenate((np.load("gt/gt_Video_45_6_4_caprabo_000151.npy"),\
#                     np.load("gt/gt_Video_45_6_4_caprabo_000201.npy")))
#gt = gt.reshape(gt.shape[0], 38, 38)

#gt = np.concatenate((np.load("gt/gt_Video_18_3_1_oreo_000001.npy"),\
#                     np.load("gt/gt_Video_18_3_1_oreo_000051.npy"),\
#                     np.load("gt/gt_Video_18_3_1_oreo_000101.npy")))
#gt = gt.reshape(gt.shape[0], 38, 38)


descriptor_mask = np.concatenate((np.load("../tensors/train/mask_Video_45_6_4_caprabo_000151.npy").squeeze(),\
                                  np.load("../tensors/train/mask_Video_45_6_4_caprabo_000201.npy").squeeze()))


cap = cv2.VideoCapture("../datasets/rrc-text-videos/ch3_train/Video_45_6_4.mp4")
#cap = cv2.VideoCapture("../datasets/rrc-text-videos/ch3_train/Video_18_3_1.mp4")
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fig, ax = plt.subplots(1, figsize=(15,15))


cap.set(cv2.CAP_PROP_POS_FRAMES, 150.);
frame = 0
for i in range(100):
    #if i <     :continue

    ret, inp = cap.read()
    if not ret:
        continue
    frame +=1

    im = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    print(cap.get(cv2.CAP_PROP_POS_FRAMES))

    plt.cla()
    ax.imshow(im)

#    resized = cv2.resize(gt[i, :, :], (int(video_width), int(video_height)), interpolation = cv2.INTER_AREA)  # gt
    resized = cv2.resize(descriptor_mask[i, :, :], (int(video_width), int(video_height)), interpolation = cv2.INTER_AREA)  # the other thing
    #resized.swapaxes(0, 1)
    ax.imshow(resized, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig('images/file%05d.jpeg' % frame, bbox_inches = 'tight', pad_inches = 0)
#    plt.show(block=False)
#    plt.pause(0.001)
plt.close()
os.system("ffmpeg -framerate 20 -pattern_type glob -i 'images/*.jpeg' -c:v mpeg4 -vb 1M -qscale:v 2 "  + "mask_desc.mp4")


