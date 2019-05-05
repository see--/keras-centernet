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
from keras_centernet.models.decode import HpDetDecode
from keras_centernet.utils.letterbox import LetterboxTransformer


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', default='output', type=str)
  parser.add_argument('--data', default='val2017', type=str)
  parser.add_argument('--annotations', default='annotations', type=str)
  parser.add_argument('--inres', default='512,512', type=str)
  parser.add_argument('--no-full-resolution', action='store_true')
  args, _ = parser.parse_known_args()
  args.inres = tuple(int(x) for x in args.inres.split(','))
  if not args.no_full_resolution:
    args.inres = (None, None)

  os.makedirs(args.output, exist_ok=True)
  kwargs = {
    'num_stacks': 2,
    'cnv_dim': 256,
    'weights': 'hpdet_coco',
    'inres': args.inres,
  }
  heads = {
    'hm': 1,  # 6
    'hm_hp': 17,  # 7
    'hp_offset': 2,  # 8
    'hps': 34,  # 9
    'reg': 2,  # 10
    'wh': 2,  # 11
  }
  out_fn_keypoints = os.path.join(args.output, args.data + '_keypoints_results_%s_%s.json' % (
      args.inres[0], args.inres[1]))
  model = HourglassNetwork(heads=heads, **kwargs)
  model = HpDetDecode(model)
  if args.no_full_resolution:
    letterbox_transformer = LetterboxTransformer(args.inres[0], args.inres[1])
  else:
    letterbox_transformer = LetterboxTransformer(mode='testing', max_stride=128)

  fns = sorted(glob(os.path.join(args.data, '*.jpg')))
  results = []
  for fn in tqdm(fns):
    img = cv2.imread(fn)
    image_id = int(os.path.splitext(os.path.basename(fn))[0])
    pimg = letterbox_transformer(img)
    pimg = normalize_image(pimg)
    pimg = np.expand_dims(pimg, 0)
    detections = model.predict(pimg)[0]
    for d in detections:
      score = d[4]
      x1, y1, x2, y2 = d[:4]
      x1, y1, x2, y2 = letterbox_transformer.correct_box(x1, y1, x2, y2)
      x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

      kps = d[5:-1]
      kps_x = kps[:17]
      kps_y = kps[17:]
      kps = letterbox_transformer.correct_coords(np.vstack([kps_x, kps_y])).T
      # add z = 1
      kps = np.concatenate([kps, np.ones((17, 1), dtype='float32')], -1)
      kps = list(map(float, kps.flatten()))

      image_result = {
        'image_id': image_id,
        'category_id': 1,
        'score': float(score),
        'bbox': [x1, y1, (x2 - x1), (y2 - y1)],
        'keypoints': kps,
      }
      results.append(image_result)

  if not len(results):
    print("No predictions were generated.")
    return

  # write output
  with open(out_fn_keypoints, 'w') as f:
    json.dump(results, f, indent=2)
  print("Predictions saved to: %s" % out_fn_keypoints)
  # load results in COCO evaluation tool
  gt_fn = os.path.join(args.annotations, 'person_keypoints_%s.json' % args.data)
  print("Loading GT: %s" % gt_fn)
  coco_true = COCO(gt_fn)
  coco_pred = coco_true.loadRes(out_fn_keypoints)
  coco_eval = COCOeval(coco_true, coco_pred, 'keypoints')
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()

  return coco_eval.stats


if __name__ == '__main__':
  main()
