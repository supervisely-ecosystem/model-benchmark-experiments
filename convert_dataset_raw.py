import json
import os


def dataset_to_coco_raw(ann_path, cocoGt: dict):
    file_names = os.listdir(ann_path)

    coco_name2id = {img['file_name']: img['id'] for img in cocoGt['images']}
    coco_cat2id = {cat['name']: cat['id'] for cat in cocoGt['categories']}

    # create coco annotations
    annotations = []
    annotation_id = 1
    for file in file_names:
        img_name = os.path.splitext(file)[0]
        img_id = coco_name2id[img_name]
        with open(os.path.join(ann_path, file), 'r') as f:
            ann = json.load(f)
        for label in ann['objects']:
            geometry_type = label['geometryType']
            if geometry_type == 'rectangle':
                class_name = label['classTitle']
                category_id = coco_cat2id[class_name]
                ((left, top), (right, bottom)) = label['points']['exterior']
                width = right - left + 1
                height = bottom - top + 1
                
                # Extract confidence score from the tag
                conf_tag = [tag for tag in label['tags'] if tag['name'] == 'confidence'][0]
                score = float(conf_tag['value'])
                
                annotation = {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [left, top, width, height],
                    "area": float(width * height),
                    "iscrowd": 0,
                    "score": score,
                }
                
                annotations.append(annotation)
                annotation_id += 1
            else:
                raise ValueError(f"Unsupported geometry type: {label.obj_class.geometry_type}")

    return annotations