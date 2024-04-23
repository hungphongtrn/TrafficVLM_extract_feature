"""Bounding box extraction for video frames."""
import json

import cv2 as cv


def get_square_box(video, bbox_file, feature_type='global'):
    """Get the square bounding box for the video.
    Args:
        video: path to video file.
        bbox_file: path to annotation file.
        feature_type: type of feature extraction, 'global', 'semi-global', 'local'.
    Returns:
        bbox_to_cut: dictionary containing frame_ids and bboxes.
    """
    if feature_type not in ['global', 'semi-global', 'local']:
        raise ValueError('feature_type must be one of "global", "semi-global", "local".')

    # prepare video dims
    cap = cv.VideoCapture(video)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # nframe = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()

    # extract bbox, with two corners, (x1,y1),(x2,y2).
    bbox_to_cut = {}

    x1, y1, x2, y2 = 1e6, 1e6, 0, 0

    try:
        with open(bbox_file) as f:  # pylint: disable=W1514
            json_data = json.load(f)['annotations']

        for bbox_data in json_data:
            x, y, w, h = bbox_data['bbox']
            x1 = int(min(x1, x))
            y1 = int(min(y1, y))
            x2 = int(max(x2, x+w))
            y2 = int(max(y2, y+h))
            frame_id = int(bbox_data['image_id'])  # used for local frames

            if feature_type == 'local':
                center_x, center_y = int((x*2+w)/2),int((y*2+w)/2)
                square_dim = max(int(w), int(h))
                xl1, xl2 = max(0, center_x-square_dim//2), min(width, center_x+square_dim//2)
                yl1, yl2 = max(0, center_y-square_dim//2), min(height, center_y+square_dim//2)
                bbox_to_cut[frame_id] = (xl1, yl1, xl2, yl2)
    # if video doesn't have corresponding annotation file, then it will be cropped to the whole image
    except FileNotFoundError:
        pass

    # for feature type not being local, then crop should be applied to the whole image
    if feature_type != 'local':
        center_x, center_y = (x1+x2)//2, (y1+y2)//2
        if feature_type == 'global':
            square_dim = min(width, height)
        else:
            square_dim = max(x2-x1, y2-y1)

        # define square crop
        if center_x-square_dim//2 < 0:
            x1 = 0
            x2 = square_dim
        elif center_x+square_dim//2 > width:
            x1 = width - square_dim
            x2 = width
        else:
            x1, x2 = center_x-square_dim//2, center_x+square_dim//2

        if center_y-square_dim//2 < 0:
            y1 = 0
            y2 = square_dim
        elif center_y+square_dim//2 > height:
            y1 = height - square_dim
            y2 = height
        else:
            y1, y2 = center_y-square_dim//2, center_y+square_dim//2
        bbox_to_cut[-1] = (x1, y1, x2, y2)
    return bbox_to_cut
