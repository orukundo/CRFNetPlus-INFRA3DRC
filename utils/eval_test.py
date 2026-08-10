# =============================================================================
# CRF-Net - INFRA-3DRC Adaptation
#
# Original implementation: CRF-Net authors/contributors
# INFRA-3DRC modifications: Olivier Rukundo
# Affiliation: University of Limerick, Ireland
#
# This file contains modifications made for training/evaluating CRF-Net
# with the INFRA-3DRC automotive perception dataset.
#
# See the repository README and original CRF-Net license for attribution.
# =============================================================================

import numpy as np
from crfnet.utils.eval import evaluate
from crfnet.utils.reporting import (
    save_overall_metrics,
    save_per_class_ap,
    generate_detection_report,
)

def evaluate_test_set(model, generator, cfg, mode='test', tensorboard=None, verbose=1):

    # run evaluation
    if cfg.distance_detection:
        best_map, best_st, best_aps, best_precisions, best_recalls, best_mean_loss_errors, best_mean_loss_errors_rel = evaluate(
            generator,
            model,
            distance=cfg.distance_detection,
            iou_threshold=0.5,
            score_threshold=0.05,
            max_detections=100,
            save_path=None,
            render=False,
            workers=cfg.workers
        )

    else:
            best_map, best_st, best_aps, best_precisions, best_recalls  = evaluate(
            generator,
            model,
            distance=cfg.distance_detection,
            iou_threshold=0.5,
            score_threshold=0.05,
            max_detections=100,
            save_path=None,
            render=False,
            workers=cfg.workers
        )
    # ignore bg with [:-1]
    mean_precision = np.nanmean(best_precisions[:-1])
    mean_recall = np.nanmean(best_recalls[:-1])


    # compute per class average precision
    total_instances = []
    precisions = []
    mean_loss_error = np.nan
    mean_loss_error_rel = np.nan
    for label, (average_precision, num_annotations ) in best_aps.items():
        if verbose == 1:
            if cfg.distance_detection:
                print('{:.0f} instances of class'.format(num_annotations),
                    generator.label_to_name(label), 'with average precision: {0:.4f} (precision: {1:.4f}, recall: {2:.4f}) and mean distance error:{3:.2f} or {4:.2f}%'\
                        .format(average_precision, best_precisions[label], best_recalls[label],best_mean_loss_errors[label], best_mean_loss_errors_rel[label]*100))
            else:
                print('{:.0f} instances of class'.format(num_annotations),
                    generator.label_to_name(label), 'with average precision: {0:.4f} (precision: {1:.4f}, recall: {2:.4f})'\
                        .format(average_precision, best_precisions[label], best_recalls[label]))
        total_instances.append(num_annotations)
        precisions.append(average_precision)

    if cfg.weighted_map:
        mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        if cfg.distance_detection and np.count_nonzero(~np.isnan(best_mean_loss_errors)):
            mean_loss_error = sum([a * b for a, b in zip(total_instances, best_mean_loss_errors) if (b==b)]) / sum(~np.isnan(best_mean_loss_errors)*total_instances)
            mean_loss_error_rel = sum([a * b for a, b in zip(total_instances, best_mean_loss_errors_rel) if (b==b)]) / sum(~np.isnan(best_mean_loss_errors_rel)*total_instances)
    else:
        mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
        if cfg.distance_detection:
            mean_loss_error = np.nanmean(best_mean_loss_errors)
            mean_loss_error_rel = np.nanmean(best_mean_loss_errors_rel)


    # TensorFlow 2 / Keras 2.11 compatibility
    if tensorboard is not None:
        import tensorflow as tf
        log_dir = getattr(tensorboard, "log_dir", None)
        if log_dir:
            writer = tf.summary.create_file_writer(log_dir)
            with writer.as_default():
                tf.summary.scalar("mAP_test_" + mode, mean_ap, step=0)
                tf.summary.scalar("precision_test_" + mode, mean_precision, step=0)
                tf.summary.scalar("recall_test_" + mode, mean_recall, step=0)
                if cfg.distance_detection:
                    tf.summary.scalar("mADE_test_" + mode, mean_loss_error, step=0)
                    tf.summary.scalar("mRDE_test_" + mode, mean_loss_error_rel, step=0)
                for label, (average_precision, num_annotations) in best_aps.items():
                    class_name = generator.label_to_name(label)
                    if class_name != "bg":
                        tf.summary.scalar(
                            f"ap_test_{mode}_{class_name} ({int(num_annotations)} instances)",
                            average_precision,
                            step=0,
                        )
            writer.flush()
            writer.close()


    # CRF-Net++ automatic experiment exports.
    results_dir = getattr(cfg, "results_dir", None)
    if results_dir:
        save_overall_metrics(
            results_dir,
            mode,
            mean_ap,
            mean_precision,
            mean_recall,
            best_st,
        )
        save_per_class_ap(
            results_dir,
            mode,
            generator,
            best_aps,
            best_precisions,
            best_recalls,
        )

        # Generate the heavy detection-level report once, for the main test set.
        if mode == "all":
            generate_detection_report(
                model,
                generator,
                cfg,
                results_dir,
                score_threshold=best_st,
                iou_threshold=0.5,
                max_detections=100,
            )

    if verbose == 1:
        print('='*60)
        print('mAP_test: {0:.4f} \t precision_test:{1:.4f} \t recall_test:{2:.4f}'.format(mean_ap, mean_precision, mean_recall))
        if cfg.distance_detection: print('mADE_test: {0:.2f} \t mRDE_test:{1:.2f}'.format(mean_loss_error, mean_loss_error_rel))
        print('@scorethreshold {0:.2f}'.format(best_st))
