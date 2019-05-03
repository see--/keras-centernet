#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import os
import time

from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
from keras_centernet.models.decode import CtDetDecode
from keras_centernet.utils.utils import COCODrawer
from keras_centernet.utils.letterbox import LetterboxTransformer


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--video', default='webcam', type=str)
  parser.add_argument('--output', default='output', type=str)
  parser.add_argument('--inres', default='512,512', type=str)
  parser.add_argument('--outres', default='1080,1920', type=str)
  parser.add_argument('--max-frames', default=1000000, type=int)
  parser.add_argument('--fps', default=24.0 * 1.25, type=float)
  args, _ = parser.parse_known_args()
  args.inres = tuple(int(x) for x in args.inres.split(','))
  args.outres = tuple(int(x) for x in args.outres.split(','))
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
  letterbox_transformer = LetterboxTransformer(args.inres[0], args.inres[1])
  cap = cv2.VideoCapture(0 if args.video == 'webcam' else args.video)
  out_fn = os.path.join(args.output, 'out.' + os.path.basename(args.video)).replace('.mp4', '.avi')
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter(out_fn, fourcc, args.fps, args.outres[::-1])
  k = 0
  tic = time.time()
  while cap.isOpened():
    if k > args.max_frames:
      print("Bye")
      break
    if k > 0 and k % 100 == 0:
      toc = time.time()
      duration = toc - tic
      print("[%05d]: %.3f seconds / 100 iterations" % (k, duration))
      tic = toc

    k += 1
    ret, img = cap.read()
    if not ret:
      print("Done")
      break
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

    out.write(img)
  print("Video saved to: %s" % out_fn)


if __name__ == '__main__':
  main()
