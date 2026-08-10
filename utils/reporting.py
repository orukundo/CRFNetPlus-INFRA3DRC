# -*- coding: utf-8 -*-
"""
CRF-Net++ evaluation/reporting utilities.

Adds research-friendly experiment artifacts without changing the detector:
- history.csv
- loss_curves.png
- map_curve.png
- per_class_AP.csv
- metrics.csv
- confusion_matrix.csv / .png
- confusion_matrix_normalized.csv / .png
- predictions.csv
- ground_truth.csv
- prediction_images/*.png

Designed for the TensorFlow 2.11 / Keras 2.11 CRF-Net port used with INFRA-3DRC.
"""

import csv
import os

import cv2
import numpy as np
import keras

from crfnet.utils.anchor_calc import compute_overlap


def _makedirs(path):
    os.makedirs(path, exist_ok=True)


def append_csv_row(path, fieldnames, row):
    """Append a row while writing the header only when the file is new."""
    _makedirs(os.path.dirname(path))
    exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_rows(path, fieldnames, rows):
    _makedirs(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_per_class_ap(results_dir, mode, generator, aps, precisions, recalls):
    path = os.path.join(results_dir, "per_class_AP.csv")

    rows = []
    for label, (ap, n_instances) in aps.items():
        class_name = generator.label_to_name(label)
        rows.append({
            "mode": mode,
            "class_id": int(label),
            "class_name": class_name,
            "instances": int(n_instances),
            "average_precision": float(ap),
            "precision": float(precisions[label]) if label < len(precisions) else np.nan,
            "recall": float(recalls[label]) if label < len(recalls) else np.nan,
        })

    # Append because eval_test is called for all/night/rain.
    fieldnames = [
        "mode", "class_id", "class_name", "instances",
        "average_precision", "precision", "recall"
    ]
    _makedirs(results_dir)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def save_overall_metrics(
    results_dir,
    mode,
    mean_ap,
    mean_precision,
    mean_recall,
    score_threshold,
):
    append_csv_row(
        os.path.join(results_dir, "metrics.csv"),
        ["mode", "mAP", "precision", "recall", "score_threshold"],
        {
            "mode": mode,
            "mAP": float(mean_ap),
            "precision": float(mean_precision),
            "recall": float(mean_recall),
            "score_threshold": float(score_threshold),
        },
    )


class TrainingReportCallback(keras.callbacks.Callback):
    """
    At training end, create loss and mAP plots from history.csv.

    CSVLogger must appear before this callback in the callback list.
    """

    def __init__(self, results_dir):
        super(TrainingReportCallback, self).__init__()
        self.results_dir = results_dir

    def on_train_end(self, logs=None):
        history_path = os.path.join(self.results_dir, "history.csv")
        if not os.path.exists(history_path):
            print("CRF-Net++: history.csv was not found; plots were skipped.")
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print("CRF-Net++: matplotlib unavailable; plots skipped:", exc)
            return

        try:
            with open(history_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            if not rows:
                return

            epochs = [int(float(r.get("epoch", i))) + 1 for i, r in enumerate(rows)]

            def values(key):
                out = []
                for r in rows:
                    try:
                        out.append(float(r[key]))
                    except Exception:
                        out.append(np.nan)
                return np.asarray(out, dtype=np.float64)

            # Loss curves
            plt.figure(figsize=(9, 6))
            found = False
            for key, label in [
                ("loss", "Training total loss"),
                ("val_loss", "Validation total loss"),
                ("regression_loss", "Training regression loss"),
                ("val_regression_loss", "Validation regression loss"),
                ("classification_loss", "Training classification loss"),
                ("val_classification_loss", "Validation classification loss"),
            ]:
                if key in rows[0]:
                    plt.plot(epochs, values(key), label=label)
                    found = True

            if found:
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("CRF-Net training and validation losses")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.results_dir, "loss_curves.png"),
                    dpi=200,
                    bbox_inches="tight",
                )
            plt.close()

            # mAP curve
            if "mAP" in rows[0]:
                plt.figure(figsize=(9, 6))
                plt.plot(epochs, values("mAP"), marker="o", markersize=3)
                plt.xlabel("Epoch")
                plt.ylabel("Validation mAP")
                plt.title("CRF-Net validation mAP")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.results_dir, "mAP_curve.png"),
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close()

            print("CRF-Net++: training history plots saved in", self.results_dir)

        except Exception as exc:
            print("CRF-Net++: could not generate training plots:", exc)


def _safe_image_for_drawing(generator, image_index):
    """
    Return a uint8 3-channel image from generator.load_image().
    INFRA-3DRC's CRF-Net adapter returns RGB/BGR + radar channels.
    """
    image = generator.load_image(image_index)
    image = np.asarray(image)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.shape[2] >= 3:
        image = image[..., :3]
    else:
        image = np.repeat(image[..., :1], 3, axis=2)

    image = image.astype(np.float32)

    # Accommodate either 0..1 or 0..255 images.
    if np.nanmax(image) <= 1.5:
        image *= 255.0

    image = np.clip(image, 0, 255).astype(np.uint8)
    return image.copy()


def _draw_box(image, box, text, thickness=2):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness)
    cv2.putText(
        image,
        text,
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _plot_confusion_matrix(matrix, labels, path, normalized=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("CRF-Net++: matplotlib unavailable; confusion plot skipped:", exc)
        return

    data = np.asarray(matrix, dtype=np.float64)
    if normalized:
        denom = data.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        data = data / denom

    fig_size = max(7, 0.8 * len(labels) + 3)
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(data, interpolation="nearest")
    plt.title("Normalized confusion matrix" if normalized else "Confusion matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.ylabel("Ground-truth class")
    plt.xlabel("Predicted class")

    threshold = np.nanmax(data) / 2.0 if data.size and np.nanmax(data) > 0 else 0.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text = "{:.2f}".format(value) if normalized else str(int(value))
            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_detection_report(
    model,
    generator,
    cfg,
    results_dir,
    score_threshold=0.05,
    iou_threshold=0.5,
    max_detections=100,
):
    """
    Generate detection-level exports and confusion matrices for the main test set.

    Detection confusion-matrix convention:
      rows    = GT classes + Background
      columns = predicted classes + Background
      GT -> Background = missed GT (false negative)
      Background -> predicted class = unmatched detection (false positive)

    Matching is greedy by descending prediction confidence and IoU >= threshold.
    Matching is class-agnostic so class-confusion errors appear off-diagonal.
    """
    _makedirs(results_dir)
    image_dir = os.path.join(results_dir, "prediction_images")
    _makedirs(image_dir)

    real_labels = [
        label
        for label in range(generator.num_classes())
        if generator.has_label(label) and generator.label_to_name(label) != "bg"
    ]
    label_to_cm = {label: i for i, label in enumerate(real_labels)}
    class_names = [generator.label_to_name(label) for label in real_labels]
    bg_idx = len(real_labels)
    cm_names = class_names + ["Background"]
    cm = np.zeros((len(cm_names), len(cm_names)), dtype=np.int64)

    prediction_rows = []
    gt_rows = []

    for image_index in range(generator.size()):
        # Model input
        inputs, _ = generator.compute_input_output([image_index])
        outputs = model.predict_on_batch(inputs)

        if len(outputs) == 4:
            boxes, scores, labels, _ = outputs
        else:
            boxes, scores, labels = outputs[:3]

        boxes = np.asarray(boxes[0])
        scores = np.asarray(scores[0])
        labels = np.asarray(labels[0]).astype(np.int32)

        keep = np.where(scores >= score_threshold)[0]
        if keep.size:
            order = keep[np.argsort(-scores[keep])[:max_detections]]
        else:
            order = np.zeros((0,), dtype=np.int64)

        pred_boxes = boxes[order]
        pred_scores = scores[order]
        pred_labels = labels[order]

        ann = generator.load_annotations(image_index)
        gt_boxes = np.asarray(ann["bboxes"], dtype=np.float32)
        gt_labels = np.asarray(ann["labels"], dtype=np.int32)

        # Export GT.
        for gt_index, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            class_name = generator.label_to_name(int(label))
            if class_name == "bg":
                continue
            gt_rows.append({
                "image_index": image_index,
                "gt_index": gt_index,
                "class_id": int(label),
                "class_name": class_name,
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
            })

        # Class-agnostic greedy matching for confusion matrix.
        gt_used = set()
        pred_matches = {}

        for pred_index in range(len(pred_boxes)):
            box = pred_boxes[pred_index]
            label = int(pred_labels[pred_index])

            best_gt = -1
            best_iou = 0.0

            if len(gt_boxes):
                overlaps = compute_overlap(
                    np.ascontiguousarray(
                        np.expand_dims(box[:4], axis=0),
                        dtype=np.float64,
                    ),
                    np.ascontiguousarray(
                        gt_boxes[:, :4],
                        dtype=np.float64,
                    ),
                )[0]

                candidate_order = np.argsort(-overlaps)
                for gt_index in candidate_order:
                    gt_index = int(gt_index)
                    if gt_index in gt_used:
                        continue
                    if float(overlaps[gt_index]) >= iou_threshold:
                        best_gt = gt_index
                        best_iou = float(overlaps[gt_index])
                        break

            if best_gt >= 0:
                gt_used.add(best_gt)
                pred_matches[pred_index] = (best_gt, best_iou)

                gt_label = int(gt_labels[best_gt])
                if gt_label in label_to_cm and label in label_to_cm:
                    cm[label_to_cm[gt_label], label_to_cm[label]] += 1
            else:
                pred_matches[pred_index] = (-1, best_iou)
                if label in label_to_cm:
                    cm[bg_idx, label_to_cm[label]] += 1

        # Missed GT -> Background.
        for gt_index, gt_label in enumerate(gt_labels):
            gt_label = int(gt_label)
            if gt_index not in gt_used and gt_label in label_to_cm:
                cm[label_to_cm[gt_label], bg_idx] += 1

        # Export predictions.
        for pred_index, (box, score, label) in enumerate(
            zip(pred_boxes, pred_scores, pred_labels)
        ):
            label = int(label)
            matched_gt, match_iou = pred_matches.get(pred_index, (-1, 0.0))
            matched_gt_class = ""
            if matched_gt >= 0:
                matched_gt_class = generator.label_to_name(
                    int(gt_labels[matched_gt])
                )

            prediction_rows.append({
                "image_index": image_index,
                "prediction_index": pred_index,
                "class_id": label,
                "class_name": generator.label_to_name(label),
                "confidence": float(score),
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "matched_gt_index": matched_gt,
                "matched_gt_class": matched_gt_class,
                "match_iou": float(match_iou),
            })

        # Prediction image: GT prefixed "GT", predictions prefixed "P".
        image = _safe_image_for_drawing(generator, image_index)

        for box, label in zip(gt_boxes, gt_labels):
            class_name = generator.label_to_name(int(label))
            if class_name != "bg":
                _draw_box(image, box, "GT:{}".format(class_name), thickness=1)

        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            _draw_box(
                image,
                box,
                "P:{} {:.2f}".format(
                    generator.label_to_name(int(label)),
                    float(score),
                ),
                thickness=2,
            )

        cv2.imwrite(
            os.path.join(image_dir, "{:06d}.png".format(image_index)),
            image,
        )

    # CSV exports.
    write_rows(
        os.path.join(results_dir, "predictions.csv"),
        [
            "image_index", "prediction_index", "class_id", "class_name",
            "confidence", "x1", "y1", "x2", "y2",
            "matched_gt_index", "matched_gt_class", "match_iou",
        ],
        prediction_rows,
    )

    write_rows(
        os.path.join(results_dir, "ground_truth.csv"),
        [
            "image_index", "gt_index", "class_id", "class_name",
            "x1", "y1", "x2", "y2",
        ],
        gt_rows,
    )

    # Raw confusion matrix CSV.
    cm_csv = os.path.join(results_dir, "confusion_matrix.csv")
    with open(cm_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["GT\\Pred"] + cm_names)
        for name, row in zip(cm_names, cm):
            writer.writerow([name] + [int(v) for v in row])

    # Normalized matrix CSV.
    denom = cm.sum(axis=1, keepdims=True).astype(np.float64)
    denom[denom == 0] = 1.0
    cm_norm = cm.astype(np.float64) / denom

    cm_norm_csv = os.path.join(
        results_dir,
        "confusion_matrix_normalized.csv",
    )
    with open(cm_norm_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["GT\\Pred"] + cm_names)
        for name, row in zip(cm_names, cm_norm):
            writer.writerow([name] + ["{:.6f}".format(v) for v in row])

    _plot_confusion_matrix(
        cm,
        cm_names,
        os.path.join(results_dir, "confusion_matrix.png"),
        normalized=False,
    )
    _plot_confusion_matrix(
        cm,
        cm_names,
        os.path.join(results_dir, "confusion_matrix_normalized.png"),
        normalized=True,
    )

    print("CRF-Net++: detection report saved in", results_dir)
