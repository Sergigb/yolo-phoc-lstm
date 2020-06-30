import xml.etree.cElementTree as ET


trans = str.maketrans({'.': r'', '"': r'', '\n': r'', '-': r'', '\'': r''})


def get_gt_from_file(gtfile):
    """
    gt_bboxes = [frame1, frame2, ..., frameN]
    frameN = [annotation1, annotation2, ..., annotationN]
    annotationN = [x, y, w, h]
    :param gtfile: path to ground truth file
    :return: gt
    """
    tree = ET.parse(gtfile)
    root = tree.getroot()

    gt_bboxes = []
    gt_ids = []
    for frame in root:
        bboxes_frame = []
        ids_frame = []
        for object_ in frame:
            if object_.get('Quality') in ('HIGH', 'MODERATE'):
                ids_frame.append(object_.get('ID'))

                x_coords = []
                y_coords = []
                for point in object_.iter('Point'):
                    x_coords.append(int(point.get('x')))
                    y_coords.append(int(point.get('y')))
                w = max(x_coords) - min(x_coords)
                h = max(y_coords) - min(y_coords)
                x = min(x_coords)
                y = min(y_coords)
                bboxes_frame.append([x, y, w, h])

        gt_bboxes.append(bboxes_frame)
        gt_ids.append(ids_frame)
    return gt_bboxes, gt_ids


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








