import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
# from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

import model_benchmark.metrics as metrics


METRIC_NAMES = {
    "mAP": "mAP",
    "precision": "Precision",
    "recall": "Recall",
    "iou": "IoU",
    "classification_accuracy": "Classification Accuracy",
    "calibration_score": "Calibration Score"
    }


def get_outcomes_per_image(matches, cocoGt: COCO):
    img_ids = sorted(cocoGt.getImgIds())
    imgId2idx = {img_id: idx for idx, img_id in enumerate(img_ids)}
    outcomes_per_image = np.zeros((len(img_ids), 3), dtype=float)
    for m in matches:
        img_id = m["image_id"]
        idx = imgId2idx[img_id]
        if m["type"] == "TP":
            outcomes_per_image[idx, 0] += 1
        elif m["type"] == "FP":
            outcomes_per_image[idx, 1] += 1
        elif m["type"] == "FN":
            outcomes_per_image[idx, 2] += 1
    return img_ids, outcomes_per_image


class MetricProvider:
    def __init__(self, eval_data: dict, cocoGt: COCO, cocoDt: COCO):

        self.cocGt = cocoGt

        # eval_data
        self.true_positives = eval_data["true_positives"]
        self.false_positives = eval_data["false_positives"]
        self.false_negatives = eval_data["false_negatives"]
        self.matches = eval_data["matches"]
        self.coco_stats = eval_data["coco_stats"]
        self.coco_precision = eval_data["coco_precision"]
        self.coco_params : Params = eval_data["coco_params"]

        # Counts
        self.TP_count = int(self.true_positives[...,0].sum())
        self.FP_count = int(self.false_positives[...,0].sum())
        self.FN_count = int(self.false_negatives[...,0].sum())

        # Matches
        self.tp_matches = [m for m in self.matches if m['type'] == "TP"]
        self.fp_matches = [m for m in self.matches if m['type'] == "FP"]
        self.fn_matches = [m for m in self.matches if m['type'] == "FN"]
        self.confused_matches = [m for m in self.fp_matches if m['miss_cls']]
        self.fp_not_confused_matches = [m for m in self.fp_matches if not m['miss_cls']]
        self.ious = np.array([m['iou'] for m in self.matches if m['iou']])

        # Calibration
        self.calibration_metrics = CalibrationMetrics(self.tp_matches, self.fp_matches, self.fn_matches, self.coco_params.iouThrs)

        # info
        self.cat_ids = cocoGt.getCatIds()
        self.cat_names = [cocoGt.cats[cat_id]['name'] for cat_id in self.cat_ids]

    def base_metrics(self):
        tp = self.true_positives.sum(1)
        fp = self.false_positives.sum(1)
        fn = self.false_negatives.sum(1)
        confuse_count = len(self.confused_matches)

        mAP = self.coco_stats[0]
        precision = np.mean(tp / (tp + fp))
        recall = np.mean(tp / (tp + fn))
        iou = np.mean(self.ious)
        classification_accuracy = self.TP_count / (self.TP_count + confuse_count)
        calibration_score = 1 - self.calibration_metrics.maximum_calibration_error()

        return {
            "mAP": mAP,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "classification_accuracy": classification_accuracy,
            "calibration_score": calibration_score
        }
    
    def per_class_metrics(self):
        tp = self.true_positives.sum(1).mean(1)
        fp = self.false_positives.sum(1).mean(1)
        fn = self.false_negatives.sum(1).mean(1)
        pr = tp / (tp + fp)
        rc = tp / (tp + fn)
        f1 = 2 * pr * rc / (pr + rc)
        return pd.DataFrame({
            "category": self.cat_names,
            "precision": pr,
            "recall": rc,
            "f1": f1
        })
    
    def pr_curve(self):
        pr_curve = self.coco_precision[:,:,:,0,2].mean(0)
        return pr_curve
    
    def prediction_table(self):
        img_ids, outcomes_per_image = get_outcomes_per_image(self.matches, self.cocoGt)
        image_names = [self.cocoGt.imgs[img_id]["file_name"] for img_id in img_ids]
        # inference_time = ...
        n_gt = outcomes_per_image[:,0] + outcomes_per_image[:,2]
        n_dt = outcomes_per_image[:,0] + outcomes_per_image[:,1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision_per_image = outcomes_per_image[:,0] / n_dt
            recall_per_image = outcomes_per_image[:,0] / n_gt
            f1_per_image = 2 * precision_per_image * recall_per_image / (precision_per_image + recall_per_image)
        prediction_table = pd.DataFrame({
            "image_name": image_names,
            "N gt": n_gt,
            "N dt": n_dt,
            "TP": outcomes_per_image[:,0],
            "FP": outcomes_per_image[:,1],
            "FN": outcomes_per_image[:,2],
            "Precision": precision_per_image,
            "Recall": recall_per_image,
            "F1": f1_per_image
            })
        return prediction_table
    
    def confusion_matrix(self):
        K = len(self.cat_ids)
        catId2idx = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        idx2catId = {i: cat_id for cat_id, i in catId2idx.items()}

        confusion_matrix = np.zeros((K+1, K+1), dtype=int)

        for m in self.confused_matches:
            cat_idx_pred = catId2idx[m['category_id']]
            cat_idx_gt = catId2idx[self.cocoGt.anns[m['gt_id']]['category_id']]
            confusion_matrix[cat_idx_pred, cat_idx_gt] += 1

        for m in self.tp_matches:
            cat_idx = catId2idx[m['category_id']]
            confusion_matrix[cat_idx, cat_idx] += 1

        for m in self.fp_not_confused_matches:
            cat_idx_pred = catId2idx[m['category_id']]
            confusion_matrix[cat_idx_pred, -1] += 1

        for m in self.fn_matches:
            cat_idx_gt = catId2idx[m['category_id']]
            confusion_matrix[-1, cat_idx_gt] += 1
        
        return confusion_matrix
    
    def frequently_confused(self, confusion_matrix, topk_pairs=20):
        # Frequently confused class pairs
        idx2catId = {i: cat_id for i, cat_id in enumerate(self.cat_ids)}
        cm = confusion_matrix[:-1,:-1]
        cm_l = np.tril(cm, -1)
        cm_u = np.triu(cm, 1)
        cm = cm_l + cm_u.T
        cm_flat = cm.flatten()
        inds_sort = np.argsort(-cm_flat)[:topk_pairs]
        inds_sort = inds_sort[cm_flat[inds_sort] > 0]  # remove zeros
        inds_sort = np.unravel_index(inds_sort, cm.shape)

        # probability of confusion: (predicted A, actually B + predicted B, actually A) / (predicted A + predicted B)
        confused_counts = cm[inds_sort]
        dt_total = confusion_matrix.sum(1)
        dt_pair_sum = np.array([dt_total[i] + dt_total[j] for i, j in zip(*inds_sort)])
        confused_prob = confused_counts / dt_pair_sum
        inds_sort2 = np.argsort(-confused_prob)

        confused_idxs = np.array(inds_sort).T[inds_sort2]
        confused_name_pairs = [(self.cat_names[i], self.cat_names[j]) for i, j in confused_idxs]
        confused_counts = confused_counts[inds_sort2]
        confused_prob = confused_prob[inds_sort2]
        confused_catIds = [(idx2catId[i], idx2catId[j]) for i, j in confused_idxs]

        return pd.DataFrame({
            "category_pair": confused_name_pairs,
            "category_id_pair": confused_catIds,
            "count": confused_counts,
            "probability": confused_prob
        })

    def iou_histogram(self):
        iou_hist = np.histogram(self.ious, range=(0.5, 1))
        return iou_hist
    
    
class CalibrationMetrics:
    def __init__(self, tp_matches, fp_matches, fn_matches, iouThrs):
        self.iouThrs = iouThrs
        scores = []
        classes = []
        iou_idxs = []
        p_matches = tp_matches + fp_matches
        per_class_count = defaultdict(int)
        for m in p_matches:
            if m['type'] == "TP" and m['iou'] is not None:
                iou_idx = np.searchsorted(iouThrs, m['iou'])
                iou_idxs.append(iou_idx)
                assert iou_idx > 0
            else:
                iou_idxs.append(0)
            scores.append(m['score'])
            classes.append(m["category_id"])
            if m['type'] == "TP":
                per_class_count[m["category_id"]] += 1
        for m in fn_matches:
            per_class_count[m["category_id"]] += 1
        per_class_count = dict(per_class_count)
        scores = np.array(scores)
        inds_sort = np.argsort(-scores)
        scores = scores[inds_sort]
        classes = np.array(classes)[inds_sort]
        iou_idxs = np.array(iou_idxs)[inds_sort]

        self.scores = scores
        self.classes = classes
        self.iou_idxs = iou_idxs
        self.per_class_count = per_class_count

        # y_true not include False Negatives, as scores for FNs can't be calculated
        self.y_true = self.iou_idxs > 0

    def scores_vs_metrics(self, iou_idx=0, cat_id=None):
        tps = self.iou_idxs > iou_idx
        if cat_id is not None:
            cls_mask = self.classes == cat_id
            tps = tps[cls_mask]
            scores = self.scores[cls_mask]
            n_positives = self.per_class_count[cat_id]
        else:
            scores = self.scores
            n_positives = sum(self.per_class_count.values())
        fps = ~tps

        tps_sum = tps.cumsum()
        fps_sum = fps.cumsum()

        # Precision, recall, f1
        precision = tps_sum / (tps_sum + fps_sum)
        recall = tps_sum / n_positives
        f1 = 2 * precision * recall / (precision + recall)
        return {
            "scores": scores,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def scores_vs_metrics_avg(self):
        res = []
        for iou_idx, iou_th in enumerate(self.iouThrs):
            metric_dict = self.scores_vs_metrics(iou_idx)
            res.append(list(metric_dict.values()))
        x = np.array(res).mean(axis=0)
        df = pd.DataFrame(x.T, columns=metric_dict.keys())
        return df

    def calibration_curve(self):
        true_probs, pred_probs = calibration_curve(self.y_true, self.scores, n_bins=10)
        return true_probs, pred_probs
    
    def maximum_calibration_error(self):
        return metrics.maximum_calibration_error(self.y_true, self.scores, n_bins=10)
    
    def scores_tp_and_fp(self, iou_idx=0):
        tps = self.iou_idxs > iou_idx
        scores_tp = self.scores[tps]
        scores_fp = self.scores[~tps]
        return scores_tp, scores_fp