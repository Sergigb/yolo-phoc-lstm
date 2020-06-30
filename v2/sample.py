import glob
import numpy as np
import torch
from model import RNN
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat

weights_path = "models/best/model-epoch-6.pth"

model = RNN(sequence_length=550)
model.cuda()
model.eval()
model.load_state_dict(torch.load(weights_path))

video_path = "../datasets/rrc-text-videos/ch3_train/Video_41_2_3.mp4"
video_name = video_path.split("/")[-1].replace(".mp4", "")

tensor_files = glob.glob("../tensors/train/tensor_" + video_name + "*")
tensor_files.sort()

tensors = None
for file in tensor_files:
    if tensors is None:
        tensors = np.load(file).squeeze()
    else:
        tensors = np.concatenate((tensors, np.load(file).squeeze()))
tensors = tensors.swapaxes(1,3).swapaxes(2,3)
tensors = torch.from_numpy(tensors).type(torch.FloatTensor).cuda()

descriptor_files = glob.glob("../tensors/train/descriptors_" + video_name + "*")
descriptor_files.sort()

descriptors = None
for file in descriptor_files:
    if descriptors is None:
        descriptors = np.load(file).squeeze()
    else:
        descriptors = np.concatenate((descriptors, np.load(file).squeeze()))

phoc_tensor = np.load("../phocs/look.npy")

descriptors_phoc = []
for i in range(descriptors.shape[0]):
    descriptors_phoc.append(np.concatenate((descriptors[i], phoc_tensor.squeeze(0))))

descriptors = np.array(descriptors_phoc)
descriptors = torch.from_numpy(descriptors).type(torch.FloatTensor).cuda()

out = model(tensors.unsqueeze(0), descriptors.unsqueeze(0))

print(out.shape)



cap = cv2.VideoCapture(video_path)
ret, inp = cap.read()

fig, ax = plt.subplots(1, figsize=(15,15))
ret, inp = cap.read()
frame = 0

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while ret:
    print(frame)
    im = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    plt.cla()
    ax.imshow(im)

    center_x, center_y, w, h, o = out[0, frame-1]

    center_x *= width
    w *= width
    center_y *= height
    h *= height
    x = center_x - w / 2.
    y = center_y - h / 2.

    rect = pat.Rectangle((x, y), abs(w), abs(h), linewidth=1, edgecolor=(1, 0, 0), facecolor='none')
    #plt.text(x1, y1, int(boxid), color=(0, 1, 0), fontsize=20)
    #plt.text(x1, y2+15, detection[1], color=(1, 0, 0), fontsize=20)
    ax.add_patch(rect)

    print(x.detach().cpu().numpy(), y.detach().cpu().numpy(), w.detach().cpu().numpy(), h.detach().cpu().numpy(), out[0, frame-1].detach().cpu().numpy())

    plt.axis('off')
    # plt.show(block=False)
    # plt.pause(0.00001)
    plt.savefig('images/file%05d.jpeg' % frame, bbox_inches = 'tight', pad_inches = 0)

    frame += 1
    ret, inp = cap.read()
plt.close()
os.system("ffmpeg -framerate 20 -pattern_type glob -i 'images/*.jpeg' -c:v mpeg4 -vb 1M -qscale:v 2 " + video_name + ".mp4")











