import math
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

coco_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',]  # noqa


def get_color(c, x, max_value, colors=[[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]):
  # https://github.com/pjreddie/darknet/blob/master/src/image.c
  ratio = (x / max_value) * 5
  i = math.floor(ratio)
  j = math.ceil(ratio)
  ratio -= i
  r = (1. - ratio) * colors[i][c] + ratio * colors[j][c]
  return r


def get_rgb_color(cls, clses):
  offset = cls * 123457 % clses
  red = get_color(2, offset, clses)
  green = get_color(1, offset, clses)
  blue = get_color(0, offset, clses)
  return int(red * 255), int(green * 255), int(blue * 255)


class COCODrawer:
  def __init__(self, width=2, font_size=22, font="assets/Roboto-Regular.ttf"):
    self.coco_names = coco_names
    self.width = width
    self.font_size = font_size
    self.font = ImageFont.truetype(font, font_size)

  def draw_box(self, img, x1, y1, x2, y2, cl):
    name = self.coco_names[cl].split()[-1]
    bgr_color = get_rgb_color(cl, len(self.coco_names))[::-1]
    # bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, self.width)
    # font background
    cv2.rectangle(img, (x1 - self.width // 2, y1 - self.font_size), (min(x1 + 70, x2 + self.width // 2), y1),
                  bgr_color, -1)
    # text
    pil_img = Image.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil_img)
    draw.text((x1 + self.width, y1 - self.font_size), name, font=self.font, fill=(0, 0, 0, 255))
    img = np.array(pil_img)[..., ::-1].copy()
    return img
