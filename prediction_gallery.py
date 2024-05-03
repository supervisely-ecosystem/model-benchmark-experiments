# per-image stats:
# many prediction count (TP+FP)
# many FP count
# many FN count
# FP + high conf (avg on image)
# many rare classes (avg)
# TP + low conf + low IoU (avg on image)


# per-instance stats:
# TP + low conf
# TP + low IoU
# FP + high conf
# rare class in GT
# FN + small area (?)
# miss-classified prediciotn + high/low conf
# miss-classified prediciotn, and it is the most often miss-classified class

import numpy as np
from pycocotools.coco import COCO


# N x k
# k = img_id, value, gt_id, dt_id
def collect(inds_sorted, values, idx2imgId, topk=100):
    return [[idx2imgId[idx], values[idx], -1, -1] for idx in inds_sorted[:topk]]

def collect_per_instance(inds_sorted, matches, key, topk=100):
    matches_sorted = [matches[idx] for idx in inds_sorted[:topk]]
    return [[m["image_id"], m[key], m["gt_id"] or -1, m["dt_id"] or -1] for m in matches_sorted]


def get_per_image(matches, cocoGt: COCO, cat_ids_rare):
    img_ids = sorted(cocoGt.getImgIds())

    # (N_imgs, 6), 6 = TP, FP, FN, score, iou, RareCls
    per_image = np.zeros((len(img_ids), 6))

    imgId2idx = {img_id: idx for idx, img_id in enumerate(img_ids)}
    idx2imgId = {idx: img_id for img_id, idx in imgId2idx.items()}

    for match in matches:
        idx = imgId2idx[match['image_id']]
        if match["type"] == "TP":
            per_image[idx, 0] += 1
            per_image[idx, 3] += match["score"]
            per_image[idx, 4] += match["iou"]
        elif match["type"] == "FP":
            per_image[idx, 1] += 1
            per_image[idx, 3] += match["score"]
        elif match["type"] == "FN":
            per_image[idx, 2] += 1
        if match["category_id"] in cat_ids_rare:
            per_image[idx, 5] += 1

    # ignore warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_image[:, 3] /= (per_image[:, 0] + per_image[:, 1])
        per_image[:, 4] /= per_image[:, 0]

    return per_image, idx2imgId


def prediction_gallery(matches, cocoGt, cat_ids_rare):
    gallery = {}
    
    ## Per-image stats
    per_image, idx2imgId = get_per_image(matches, cocoGt, cat_ids_rare)

    # many FP count
    fp_count = per_image[:, 1]
    inds_sorted = np.argsort(fp_count)[::-1]
    gallery["many_FP_img"] = collect(inds_sorted, fp_count, idx2imgId)

    # many FN count
    fn_count = per_image[:, 2]
    inds_sorted = np.argsort(fn_count)[::-1]
    gallery["many_FN_img"] = collect(inds_sorted, fn_count, idx2imgId)

    # FP + high conf (avg on image), log-score
    log_scores = np.log(1 + per_image[:, 1]) * per_image[:, 3]
    log_scores = np.nan_to_num(log_scores, nan=-1)
    inds_sorted = np.argsort(log_scores)[::-1]
    gallery["FP_high_conf_img"] = collect(inds_sorted, log_scores, idx2imgId)

    # low conf
    conf = per_image[:, 3]
    inds_sorted = np.argsort(per_image[:, 3])
    gallery["low_conf_img"] = collect(inds_sorted, conf, idx2imgId)

    # low IoU
    iou = per_image[:, 4]
    inds_sorted = np.argsort(per_image[:, 4])
    gallery["low_iou_img"] = collect(inds_sorted, iou, idx2imgId)

    # many rare classes
    rare_cls = per_image[:, 5]
    inds_sorted = np.argsort(per_image[:, 5])[::-1]
    gallery["many_rare_cls_img"] = collect(inds_sorted, rare_cls, idx2imgId)


    ## Per-instance stats
    tp_matches = [match for match in matches if match["type"] == "TP"]
    fp_matches = [match for match in matches if match["type"] == "FP" and not match["miss_cls"]]
    # fn_matches = [match for match in matches if match["type"] == "FN"]
    confused_matches = [match for match in matches if match["miss_cls"]]

    # TP + low conf
    inds_sorted = np.argsort([match["score"] for match in tp_matches])
    gallery["TP_low_conf"] = collect_per_instance(inds_sorted, tp_matches, "score")

    # TP + low IoU
    inds_sorted = np.argsort([match["iou"] for match in tp_matches])
    gallery["TP_low_iou"] = collect_per_instance(inds_sorted, tp_matches, "iou")

    # FP + high conf
    inds_sorted = np.argsort([match["score"] for match in fp_matches])[::-1]
    gallery["FP_high_conf"] = collect_per_instance(inds_sorted, fp_matches, "score")

    # confused + high/low conf
    inds_sorted = np.argsort([match["score"] for match in confused_matches])[::1]
    gallery["confused_high_conf"] = collect_per_instance(inds_sorted, confused_matches, "score")

    inds_sorted = inds_sorted[::-1]
    gallery["confused_low_conf"] = collect_per_instance(inds_sorted, confused_matches, "score")

    return gallery
