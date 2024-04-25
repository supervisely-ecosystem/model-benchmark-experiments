import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageDraw, ImageFont


font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)


def get_counts(cocoEval: COCOeval):
    """
    true_positives, false_positives, false_negatives, scores
    """
    aRng = cocoEval.params.areaRng[0]
    eval_imgs = [ev for ev in cocoEval.evalImgs if ev is not None and ev['aRng'] == aRng]

    N = len(eval_imgs)
    T = len(cocoEval.params.iouThrs)
    K = len(cocoEval.params.catIds)

    true_positives = np.zeros((K, N, T))
    false_positives = np.zeros((K, N, T))
    false_negatives = np.zeros((K, N, T))
    scores = []
    # n_positives = np.zeros(K)

    for i, eval_img in enumerate(eval_imgs):

        catId = eval_img['category_id']
        imgId = eval_img['image_id']
        
        true_positives[catId-1, i] = np.sum(eval_img['dtMatches'] > 0, axis=1)
        false_positives[catId-1, i] = np.sum(eval_img['dtMatches'] == 0, axis=1)
        false_negatives[catId-1, i] = np.sum(eval_img['gtMatches'] == 0, axis=1)
        scores.append(eval_img['dtScores'])
        # n_positives[catId-1] += eval_img['gtMatches'].shape[1]

    return true_positives, false_positives, false_negatives, scores


def get_counts_and_scores(cocoEval: COCOeval, cat_id: int, t: int):
    """
    tps, fps, scores, n_positives
    """
    aRng = cocoEval.params.areaRng[0]
    eval_imgs = [ev for ev in cocoEval.evalImgs if ev is not None and ev['aRng'] == aRng]

    tps = []
    fps = []
    # fns = []
    scores = []
    n_positives = 0

    # Process each evaluated image
    for eval_img in eval_imgs:
        if eval_img['category_id'] != cat_id:
            continue
        dtScores = eval_img['dtScores']
        dtm = eval_img['dtMatches'][t]
        gtm = eval_img['gtMatches'][t]

        # ntp = (dtm > 0).sum()
        # nfp = (dtm == 0).sum()
        # nfn = (gtm == 0).sum()
        p = len(gtm)

        tp = (dtm > 0).astype(int).tolist()
        fp = (dtm == 0).astype(int).tolist()
        # fn = [nfn]*len(dtm)

        tps.extend(tp)
        fps.extend(fp)
        # fns.extend(fn)
        scores.extend(dtScores)
        n_positives += p

    assert len(tps) == len(fps) == len(scores)

    # sort by score
    indices = np.argsort(scores)[::-1]
    scores = np.array(scores)[indices]
    tps = np.array(tps)[indices]
    fps = np.array(fps)[indices]

    return tps, fps, scores, n_positives


def show_gt_image(img_id, cocoGt: COCO, prefix="./data/COCO2017/img/val2017/", draw_label=True):
    ann_ids = cocoGt.getAnnIds([img_id])
    anns = cocoGt.loadAnns(ann_ids)
    img = cocoGt.loadImgs([img_id])[0]
    im = Image.open(prefix+img["file_name"])
    for ann in anns:
        bbox = ann["bbox"]
        x, y, w, h = bbox
        draw = ImageDraw.Draw(im)
        draw.rectangle([x, y, x+w, y+h], outline="red", width=5)
        # class name
        cat_id = ann["category_id"]
        class_name = cocoGt.cats[cat_id]["name"]
        if draw_label:
            draw.text((x, y), class_name, fill="black", font=font)
    return im


def show_pred_image(img_id, cocoDt: COCO, prefix="./data/COCO2017/img/val2017/", draw_label=True):
    ann_ids = cocoDt.getAnnIds([img_id])
    anns = cocoDt.loadAnns(ann_ids)
    img = cocoDt.loadImgs([img_id])[0]
    im = Image.open(prefix+img["file_name"])
    for ann in anns:
        bbox = ann["bbox"]
        x, y, w, h = bbox
        draw = ImageDraw.Draw(im)
        draw.rectangle([x, y, x+w, y+h], outline="blue", width=5)
        # class name
        cat_id = ann["category_id"]
        class_name = cocoDt.cats[cat_id]["name"]
        if draw_label:
            draw.text((x, y), class_name, fill="black", font=font)
    return im
