"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
import numpy as np

from .colors import label_color, tum_colors
import pprint




def visualize_predictions(predictions, image_data_vis, generator, dist=False, verbose=False, cfg=None):
    """
    Visualizes predictions as bounding boxes with readable, collision-aware labels.

    Detector outputs are unchanged; only caption placement is improved.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    box_thickness = 2
    padding = 2
    vertical_gap = 2

    all_dets = []
    [bboxes, probs, labels] = predictions

    image_h, image_w = image_data_vis.shape[:2]
    occupied_label_boxes = []

    def overlaps(a, b):
        return not (
            a[2] <= b[0] or a[0] >= b[2] or
            a[3] <= b[1] or a[1] >= b[3]
        )

    def find_label_position(box_x1, box_y1, text_w, text_h, baseline):
        label_w = text_w + 2 * padding
        label_h = text_h + baseline + 2 * padding

        x = int(round(box_x1))
        x = max(0, min(image_w - label_w, x))

        preferred_top = int(round(box_y1)) - label_h - vertical_gap
        candidate_tops = []

        for k in range(8):
            candidate_tops.append(preferred_top - k * (label_h + vertical_gap))

        below_start = int(round(box_y1)) + vertical_gap
        for k in range(8):
            candidate_tops.append(below_start + k * (label_h + vertical_gap))

        for top in candidate_tops:
            top = max(0, min(image_h - label_h, top))
            candidate = [x, top, x + label_w, top + label_h]
            if not any(overlaps(candidate, used) for used in occupied_label_boxes):
                occupied_label_boxes.append(candidate)
                return candidate

        top = max(0, min(image_h - label_h, preferred_top))
        candidate = [x, top, x + label_w, top + label_h]
        occupied_label_boxes.append(candidate)
        return candidate

    for jk in range(bboxes.shape[1]):
        x1, y1, x2, y2 = bboxes[0, jk, :]
        x1, y1, x2, y2 = [int(round(float(v))) for v in (x1, y1, x2, y2)]

        x1 = max(0, min(image_w - 1, x1))
        x2 = max(0, min(image_w - 1, x2))
        y1 = max(0, min(image_h - 1, y1))
        y2 = max(0, min(image_h - 1, y2))

        key = generator.label_to_name(labels[0, jk])
        color = tuple(int(v) for v in (tum_colors[key] * 255))

        cv2.rectangle(
            image_data_vis,
            (x1, y1),
            (x2, y2),
            color,
            box_thickness,
            cv2.LINE_AA,
        )

        if dist is not False:
            text_label = "{0}: {1:3.1f} {2}".format(
                key.split(".", 1)[-1],
                dist[0, jk],
                "m",
            )
            all_dets.append((key, 100 * probs[0, jk], dist[0, jk]))
        else:
            text_label = "{}: {:.2f}".format(
                key.split(".", 1)[-1],
                float(probs[0, jk]),
            )
            all_dets.append((key, 100 * probs[0, jk]))

        (text_w, text_h), baseline = cv2.getTextSize(
            text_label,
            font,
            font_scale,
            font_thickness,
        )

        lx1, ly1, lx2, ly2 = find_label_position(
            x1,
            y1,
            text_w,
            text_h,
            baseline,
        )

        cv2.rectangle(
            image_data_vis,
            (lx1, ly1),
            (lx2, ly2),
            color,
            -1,
        )

        text_x = lx1 + padding
        text_y = ly2 - baseline - padding

        cv2.putText(
            image_data_vis,
            text_label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    if verbose:
        pprint.pprint(all_dets)

    return image_data_vis

def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """Draw a readable caption near a bounding box."""
    b = np.array(box).astype(int)
    h, w = image.shape[:2]

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    thickness = 1
    padding = 2

    (text_w, text_h), baseline = cv2.getTextSize(
        caption, font, font_scale, thickness
    )

    label_w = text_w + 2 * padding
    label_h = text_h + baseline + 2 * padding

    x1 = max(0, min(w - label_w, b[0]))
    y1 = b[1] - label_h - 2

    if y1 < 0:
        y1 = max(0, min(h - label_h, b[1] + 2))

    x2 = min(w - 1, x1 + label_w)
    y2 = min(h - 1, y1 + label_h)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

    cv2.putText(
        image,
        caption,
        (x1 + padding, y2 - baseline - padding),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, boxes, scores, labels, dist=False, color=None, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        if dist is False:
            caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        else:
            caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}   {1:.2f}m'.format(scores[i], dist[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary containing bboxes (shaped [N, 4]) and labels (shaped [N]).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert('bboxes' in annotations)
    assert('labels' in annotations)
    assert(annotations['bboxes'].shape[0] == annotations['labels'].shape[0])

    for i in range(annotations['bboxes'].shape[0]):
        label   = annotations['labels'][i]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, annotations['bboxes'][i], caption)
        draw_box(image, annotations['bboxes'][i], color=c)
