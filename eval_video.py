import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat

from utils import Sampler


# video_path = 'datasets/rrc-text-videos/ch3_test/Video_35_2_3.mp4'
video_path = 'datasets/rrc-text-videos/ch3_test/Video_49_6_4.mp4'
# video_path = 'datasets/rrc-text-videos/ch3_train/Video_45_6_4.mp4'
video_name = video_path.split('/')[-1].replace('.mp4', '')
# descriptors_path = 'extracted_descriptors_100'
descriptors_path = 'extracted_descriptors_100_test'
# weights_path = 'models/model-epoch-200.pth'
weights_path = 'models/best/model-epoch-last.pth'

queries = [b'aditivos']
# queries = [b'caprabo']

cap = cv2.VideoCapture(video_path)
sampler = Sampler(weights_path=weights_path)

for word in queries:
    print('query: ' + word.decode('utf-8'))

    predictions_loc, predictions = sampler.sample_video(word.decode('utf-8'), video_name, print_sorted_files=True,
                                                        descriptors_path=descriptors_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ret, inp = cap.read()

    frame = 0
    # for i in range(100):  # skip frames if necessary
    #     frame += 1
    #     ret, inp = cap.read()

    fig, ax = plt.subplots(1)
    while ret:
        im = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        center_x, center_y, w, h = predictions_loc[0, frame, :]
        center_x *= inp.shape[1]
        print(inp.shape[1], inp.shape[0])
        w *= inp.shape[1]
        center_y *= inp.shape[0]
        h *= inp.shape[0]

        objectness = predictions[0, frame]
        x = center_x - w / 2.
        y = center_y - h / 2.
        print(x.item(), y.item(), abs(w.item()), abs(h.item()), objectness.item())

        rect = pat.Rectangle((x, y), abs(w), abs(h), linewidth=1, edgecolor=(objectness.item(), 0, 0), facecolor='none')
        plt.cla()
        ax.imshow(im)
        ax.add_patch(rect)
        plt.text(40, 80, str('%1.2f' % objectness.item()), color=(objectness.item(), 0, 0), fontsize=20)
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.00001)
        # plt.savefig('images/file%05d.png' % frame)

        frame += 1
        ret, inp = cap.read()



