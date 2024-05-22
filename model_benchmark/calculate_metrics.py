from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from model_benchmark import utils


def calculate_metrics(cocoGt: COCO, cocoDt: COCO):
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    # For classification metrics
    cocoEval_cls = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval_cls.params.useCats = 0
    cocoEval_cls.evaluate()
    cocoEval_cls.accumulate()
    cocoEval_cls.summarize()

    true_positives, false_positives, false_negatives = utils.get_counts(cocoEval)
    eval_img_dict = utils.get_eval_img_dict(cocoEval)
    eval_img_dict_cls = utils.get_eval_img_dict(cocoEval_cls)
    matches = utils.get_matches(eval_img_dict, eval_img_dict_cls, cocoEval_cls, iou_t=0)

    eval_data = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matches": matches,
        "coco_stats": cocoEval.stats,
        "coco_precision": cocoEval.eval['precision'],
        "coco_params": cocoEval.params,
    }
    
    return eval_data