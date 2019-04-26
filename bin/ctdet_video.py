import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

from lib.models.networks.hourglass import HourglassNetwork
from lib.models.decode import CtDetDecode

np.random.seed(123)
num_classes = 80
COLORS = [tuple(int(x) for x in ((np.random.random((3, )) * 0.6 + 0.4) * 255).astype(np.uint8))
          for _ in range(num_classes)]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--video', default='webcam', type=str, help='Path to video file or `webcam`')
  args, _ = parser.parse_known_args()
  kwargs = {
      'num_stacks': 2,
      'cnv_dim': 256,
      'weights': 'coco',
      'inres': (512, 512),
  }
  heads = {
    'hm': 80,
    'reg': 2,
    'wh': 2
  }
  model = HourglassNetwork(heads=heads, **kwargs)
  model = CtDetDecode(model)
  img = cv2.imread(args.fn)
  img = cv2.resize(img, kwargs['inres'][::-1])
  pimg = np.float32(img) / 255.
  pimg = (pimg - [0.408, 0.447, 0.470]) / [0.289, 0.274, 0.278]
  pimg = np.expand_dims(pimg, 0)

  detections = model.predict(pimg)[0]

  for d in detections[:10]:
    x1, y1, x2, y2, score, cl = d
    if score < 0.001:
      break
    x1, y1, x2, y2 = int(x1.round()), int(y1.round()), int(x2.round()), int(y2.round())
    color = COLORS[int(cl)]
    print(color)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    print(x1, y1, x2, y2, score, cl)

  out_fn = os.path.basename(os.path.join('/tmp', 'out.' + os.path.basename(args.fn)))
  cv2.imwrite(out_fn, img)
  print("Image saved to: %s" % out_fn)

  plt.imshow(img[..., ::-1])
  plt.show()


if __name__ == '__main__':
  main()
