import argparse
import cv2
import numpy as np
import os

from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
from keras_centernet.models.decode import CtDetDecode
from keras_centernet.utils import get_rgb_color, letterbox_image, invert_transform


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
  img = cv2.imread(args.fn)
  pimg, M = letterbox_image(img, args.inres[0], args.inres[1])
  M_inv = invert_transform(M)
  pimg = normalize_image(pimg)
  pimg = np.expand_dims(pimg, 0)
  detections = model.predict(pimg)[0]
  for d in detections:
    x1, y1, x2, y2, score, cl = d
    if score < 0.001:
      break
    pnts = np.float32([[x1, x2], [y1, y2]])
    pnts = M_inv[:2, :2] @ (pnts + M_inv[:, 2:3])
    x1, y1, x2, y2 = pnts[0, 0], pnts[1, 0], pnts[0, 1], pnts[1, 1]
    x1, y1, x2, y2 = int(x1.round()), int(y1.round()), int(x2.round()), int(y2.round())
    color = get_rgb_color(int(cl), heads['hm'])[::-1]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

  out_fn = os.path.join(args.output, 'out.' + os.path.basename(args.fn))
  cv2.imwrite(out_fn, img)
  print("Image saved to: %s" % out_fn)


if __name__ == '__main__':
  main()
