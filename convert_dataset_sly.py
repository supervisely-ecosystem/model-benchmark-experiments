import supervisely as sly


def create_category_id_map(meta: sly.ProjectMeta, coco_categories: list):
    name2id = {category['name']: category['id'] for category in coco_categories}
    category_id_map = {}
    for i, obj_class in enumerate(meta.obj_classes):
        category_id_map[obj_class.name] = name2id[obj_class.name]
    return category_id_map


def dataset_to_coco(dataset: sly.Dataset, meta: sly.ProjectMeta, cocoGt: dict):
    category_id_map = create_category_id_map(meta, cocoGt['categories'])
    # img_name -> img_id
    dt_img_names = dataset.get_items_names()
    coco_name2id = {img['file_name']: img['id'] for img in cocoGt['images']}
    dt_name2id = {img_name: coco_name2id[img_name] for img_name in dt_img_names}

    # create coco annotations
    coco_dt = {
        "annotations": []
    }
    annotation_id = 1
    for img_name, img_id in dt_name2id.items():
        ann : sly.Annotation = dataset.get_ann(img_name, meta)
        for label in ann.labels:
            geometry_type = label.obj_class.geometry_type
            if geometry_type in [sly.Rectangle, sly.AnyGeometry]:
                category_id = category_id_map[label.obj_class.name]
                bbox : sly.Rectangle = label.geometry.to_bbox()
                
                # Extract confidence score from the tag
                score = float(label.tags.get("confidence").value)
                
                annotation = {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [bbox.left, bbox.top, bbox.width, bbox.height],
                    "area": float(bbox.width * bbox.height),
                    "iscrowd": 0,
                    "score": score,
                }
                
                coco_dt["annotations"].append(annotation)
                annotation_id += 1
            else:
                raise ValueError(f"Unsupported geometry type: {label.obj_class.geometry_type}")

    return coco_dt
