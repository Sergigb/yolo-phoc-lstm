from glob import glob

import motmetrics as mm
import cv2

from utils import get_gt_from_file, Sampler

trans = str.maketrans({'.': r'', '"': r'', '\n': r'', '-': r'', '\'': r''})
threshold = 0.75

def get_metrics(hypothesis_bboxes, hypothesis_ids, gt_path, acc):
    """
    :param hypothesis_bboxes:
    :param hypothesis_ids:
    :param gt_path: path to the gt file
    :return: nothingness
    """
    gt_bboxes, gt_ids = get_gt_from_file(gt_path)
    # print(len(gt_bboxes), len(gt_ids), len(hypothesis_bboxes), len(hypothesis_ids))

    for gt_bboxes_frame, gt_ids_frame, hyp_bboxes_frame, hyp_ids_frame in zip(gt_bboxes, gt_ids,
                                                                              hypothesis_bboxes, hypothesis_ids):
        distances_frame = mm.distances.iou_matrix(gt_bboxes_frame, hyp_bboxes_frame, max_iou=0.5)
        acc.update(gt_ids_frame, hyp_ids_frame, distances_frame)


if __name__ == '__main__':
    gt_path = 'datasets/rrc-text-videos/ch3_test/'
    descriptors_path = 'extracted_descriptors_100_test'
    annotations_paths = glob(gt_path + '*.xml')
    sampler = Sampler(weights_path='models/best/model-epoch-last.pth')
    acc = mm.MOTAccumulator(auto_id=True)
    for annotations_path in annotations_paths:
        print("Evaluating file ", annotations_path)
        video_path = annotations_path.replace("_GT.xml", ".mp4")
        video_name = video_path.split('/')[-1].replace('.mp4', '')
        voc_path = annotations_path.replace("GT.xml", "GT_voc.txt")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        predicted_bboxes = [[] for dim in range(num_frames)]
        predicted_ids = [[] for dim in range(num_frames)]

        with open(voc_path) as f:
            lines = f.readlines()

        for line in lines:
            # line = line.split(',')[-1]
            word = line.translate(trans).lower()
            predictions_loc, predictions = sampler.sample_video(word, video_name, descriptors_path=descriptors_path)
            predictions_loc = predictions_loc.squeeze(0)
            predictions = predictions.squeeze(0).squeeze(1)

            for i in range(num_frames):
                if predictions[i] > threshold:
                    center_x, center_y, w, h = predictions_loc[i, :].tolist()
                    center_x *= width
                    w *= width
                    center_y *= height
                    h *= height

                    x = center_x - w / 2.
                    y = center_y - h / 2.

                    predicted_bboxes[i].append([x, y, w, h])
                    predicted_ids[i].append(word)  # the id is the word since we have no way of telling detections apart

        get_metrics(predicted_bboxes, predicted_ids, annotations_path, acc)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        [acc, acc.events.loc[0:1]],
        metrics=mm.metrics.motchallenge_metrics,
        names=['full', 'part'])

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names)

    print(strsummary)

