#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
from glob import glob

from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
from keras_centernet.models.decode import HpDetDecode
from keras_centernet.utils.utils import COCODrawer
from keras_centernet.utils.letterbox import LetterboxTransformer


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fn', default='assets/demo.jpg', type=str)
  parser.add_argument('--output', default='output', type=str)
  parser.add_argument('--inres', default='512,512', type=str)
  args, _ = parser.parse_known_args()
  args.inres = tuple(int(x) for x in args.inres.split(','))
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
  model = HourglassNetwork(heads=heads, **kwargs)
  model = HpDetDecode(model)
  drawer = COCODrawer()
  fns = sorted(glob(args.fn))
  for fn in tqdm(fns):
    img = cv2.imread(fn)
    letterbox_transformer = LetterboxTransformer(args.inres[0], args.inres[1])
    pimg = letterbox_transformer(img)
    pimg = normalize_image(pimg)
    pimg = np.expand_dims(pimg, 0)
    detections = model.predict(pimg)[0]
    for d in detections:
      score, cl = d[4], d[-1]
      if score < 0.3:
        break
      x1, y1, x2, y2 = d[:4]
      kps = d[5:-1]
      kps_x = kps[:17]
      kps_y = kps[17:]
      kps = letterbox_transformer.correct_coords(np.vstack([kps_x, kps_y])).T
      x1, y1, x2, y2 = letterbox_transformer.correct_box(x1, y1, x2, y2)
      img = drawer.draw_pose(img, kps)
      img = drawer.draw_box(img, x1, y1, x2, y2, cl)

    out_fn = os.path.join(args.output, 'hpdet.' + os.path.basename(fn))
    cv2.imwrite(out_fn, img)
    print("Image saved to: %s" % out_fn)


if __name__ == '__main__':
  main()
