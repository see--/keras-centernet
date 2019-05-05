#!/usr/bin/env python3

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np  # noqa

import json

tmp_fn = "/tmp/tmp.json"
results_fn = "output/val2017_bbox_results_512_512.json"
results = json.load(open(results_fn))
results = [r for r in results if r['score'] > 0.3]
json.dump(results, open(tmp_fn, 'w'))

# image_ids = [r["image_id"] for r in results]
coco_true = COCO("annotations/instances_val2017.json")
coco_pred = coco_true.loadRes(tmp_fn)

# run COCO evaluation
coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
