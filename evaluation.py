from glob import glob

import motmetrics as mm
import cv2

from utils import get_gt_from_file, Sampler, trans, load_descriptors
from nms import nms
threshold = 0.90

def get_metrics(hypothesis_bboxes, hypothesis_ids, gt_path, acc):
    """
    :param hypothesis_bboxes:
    :param hypothesis_ids:
    :param gt_path: path to the gt file
    :return: nothingness
    """
    gt_bboxes, gt_ids = get_gt_from_file(gt_path)

    for gt_bboxes_frame, gt_ids_frame, hyp_bboxes_frame, hyp_ids_frame in zip(gt_bboxes, gt_ids,
                                                                              hypothesis_bboxes, hypothesis_ids):
        distances_frame = mm.distances.iou_matrix(gt_bboxes_frame, hyp_bboxes_frame)
        acc.update(gt_ids_frame, hyp_ids_frame, distances_frame)

gt_path = 'datasets/rrc-text-videos/ch3_test/'  # test
# gt_path = 'datasets/rrc-text-videos/ch3_train/'  # train
descriptors_path = 'extracted_descriptors/extracted_descriptors_50_test'  # test
# descriptors_path = 'extracted_descriptors/extracted_descriptors_50'  # train
annotations_paths = glob(gt_path + '*.xml')
# sampler = Sampler(weights_path='models/best/model-epoch-last.pth')
num_descriptors = 50
sampler = Sampler(weights_path='models/model-epoch-last.pth', num_descriptors=num_descriptors, hidden_size=512)
acc = mm.MOTAccumulator(auto_id=True)

for annotations_path in annotations_paths:
    print("Processing file ", annotations_path)
    video_path = annotations_path.replace("_GT.xml", ".mp4")
    video_name = video_path.split('/')[-1].replace('.mp4', '')
    voc_path = annotations_path.replace("GT.xml", "GT_voc.txt")  # test
    # voc_path = annotations_path.replace("GT.xml", "GT.txt")  # train

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    predicted_bboxes = [[] for dim in range(num_frames)]
    predicted_ids = [[] for dim in range(num_frames)]

    with open(voc_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.split(',')[-1]
        word = line.translate(trans).lower()
        predictions = sampler.sample_video(word, video_name, descriptors_path=descriptors_path)
        predictions = predictions.cpu().squeeze(0)

        descriptors = load_descriptors(video_name, word, descriptors_path, num_descriptors=num_descriptors)
        for frame in range(num_frames):
            predicted_bboxes_frame = []
            predicted_ids_frame = []
            predicted_scores = []
            for j in range(descriptors.shape[1]):
                if predictions[frame, j] > 0.5:
                    center_x, center_y, w, h, p, _ = descriptors[frame, j]
                    center_x *= width
                    w *= width
                    center_y *= height
                    h *= height
                    x = center_x - w / 2.
                    y = center_y - h / 2.
                    predicted_bboxes_frame.append([x, y, w, h])
                    predicted_ids_frame.append(word)
                    predicted_scores.append(predictions[frame, j])

            indices = nms.boxes(predicted_bboxes_frame, predicted_scores)  # non maximal suppresion
            for index in indices:
                predicted_bboxes[frame].append(predicted_bboxes_frame[index])
                predicted_ids[frame].append(predicted_ids_frame[index])

    get_metrics(predicted_bboxes, predicted_ids, annotations_path, acc)
mh = mm.metrics.create()
summary = mh.compute_many(
    [acc],
    metrics=mm.metrics.motchallenge_metrics,
    names=['full'])

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names)

print(strsummary)

