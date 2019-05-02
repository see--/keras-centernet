#!/usr/bin/env python3
import argparse
import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
import os
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
from keras_centernet.models.decode import CtDetDecode
from keras_centernet.utils.letterbox import LetterboxTransformer

# https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/datasets/coco.py
COCO_IDS = [0,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
            74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', default='output', type=str)
  parser.add_argument('--data', default='val2017', type=str)
  parser.add_argument('--annotations', default='annotations', type=str)
  parser.add_argument('--inres', default='512,512', type=str)
  args, _ = parser.parse_known_args()
  args.inres = tuple(int(x) for x in args.inres.split(','))
  os.makedirs(args.output, exist_ok=True)
  kwargs = {
    'num_stacks': 2,
    'cnv_dim': 256,
    'weights': 'coco',
    'inres': args.inres,
  }
  heads = {
    'hm': 80,
    'reg': 2,
    'wh': 2
  }
  out_fn_box = os.path.join(args.output, args.data + '_bbox_results_%s_%s.json' % (args.inres[0], args.inres[1]))
  out_fn_image_ids = os.path.join(args.output, args.data + '_processed_image_ids_%s_%s.json' % (
    args.inres[0], args.inres[1]))

  model = HourglassNetwork(heads=heads, **kwargs)
  model = CtDetDecode(model)
  letterbox_transformer = LetterboxTransformer(args.inres[0], args.inres[1])
  fns = sorted(glob(os.path.join(args.data, '*.jpg')))
  results = []
  image_ids = []
  for fn in tqdm(fns):
    img = cv2.imread(fn)
    image_id = int(os.path.splitext(os.path.basename(fn))[0])
    pimg = letterbox_transformer(img)
    pimg = normalize_image(pimg)
    pimg = np.expand_dims(pimg, 0)
    detections = model.predict(pimg)[0]
    for d in detections:
      x1, y1, x2, y2, score, cl = d
      # if score < 0.001:
      #   break
      x1, y1, x2, y2 = letterbox_transformer.correct_box(x1, y1, x2, y2)
      x1, y1, x2, y2, cl = int(x1.round()), int(y1.round()), int(x2.round()), int(y2.round()), int(cl)
      image_result = {
        'image_id': image_id,
        'category_id': COCO_IDS[cl + 1],
        'score': float(score),
        'bbox': [x1, y1, (x2 - x1), (y2 - y1)],
      }
      results.append(image_result)
    image_ids.append(image_id)

  if not len(results):
    print("No predictions were generated.")
    return

  # write output
  with open(out_fn_box, 'w') as f:
    json.dump(results, f, indent=2)
  with open(out_fn_image_ids, 'w') as f:
    json.dump(image_ids, f, indent=2)
  print("Predictions saved to: %s & %s" % (out_fn_box, out_fn_image_ids))
  # load results in COCO evaluation tool
  gt_fn = os.path.join(args.annotations, 'instances_%s.json' % args.data)
  print("Loading GT: %s" % gt_fn)
  coco_true = COCO(gt_fn)
  coco_pred = coco_true.loadRes(out_fn_box)

  # run COCO evaluation
  coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
  coco_eval.params.imgIds = image_ids
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()
  return coco_eval.stats


if __name__ == '__main__':
  main()
