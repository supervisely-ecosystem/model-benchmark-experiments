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

    eval_img_dict = utils.get_eval_img_dict(cocoEval)
    eval_img_dict_cls = utils.get_eval_img_dict(cocoEval_cls)
    matches = utils.get_matches(eval_img_dict, eval_img_dict_cls, cocoEval_cls, iou_t=0)

    params = {'iouThrs': cocoEval.params.iouThrs, 'recThrs': cocoEval.params.recThrs}
    coco_metrics = {'mAP': cocoEval.stats[0], 'precision': cocoEval.eval['precision']}
    eval_data = {
        "matches": matches,
        "coco_metrics": coco_metrics,
        "params": params,
    }
    
    return eval_data