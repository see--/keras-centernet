#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import os

from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
from keras_centernet.models.decode import CtDetDecode
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
      'weights': 'coco',
      'inres': args.inres,
  }
  heads = {
    'hm': 80,
    'reg': 2,
    'wh': 2
  }
  model = HourglassNetwork(heads=heads, **kwargs)
  model = CtDetDecode(model)
  drawer = COCODrawer()
  img = cv2.imread(args.fn)
  letterbox_transformer = LetterboxTransformer(args.inres[0], args.inres[1])
  pimg = letterbox_transformer(img)
  pimg = normalize_image(pimg)
  pimg = np.expand_dims(pimg, 0)
  detections = model.predict(pimg)[0]
  for d in detections:
    x1, y1, x2, y2, score, cl = d
    if score < 0.3:
      break
    x1, y1, x2, y2 = letterbox_transformer.correct_box(x1, y1, x2, y2)
    x1, y1, x2, y2, cl = int(x1.round()), int(y1.round()), int(x2.round()), int(y2.round()), int(cl)
    img = drawer.draw_box(img, x1, y1, x2, y2, cl)

  out_fn = os.path.join(args.output, 'out.' + os.path.basename(args.fn))
  cv2.imwrite(out_fn, img)
  print("Image saved to: %s" % out_fn)


if __name__ == '__main__':
  main()
