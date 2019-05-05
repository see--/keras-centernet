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
  def __init__(self, font_size=24, font="assets/Roboto-Regular.ttf", char_width=14):
    self.coco_names = coco_names
    self.font_size = font_size
    self.font = ImageFont.truetype(font, font_size)
    self.char_width = char_width

    self.num_joints = 17
    self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                  [3, 5], [4, 6], [5, 6],
                  [5, 7], [7, 9], [6, 8], [8, 10],
                  [5, 11], [6, 12], [11, 12],
                  [11, 13], [13, 15], [12, 14], [14, 16]]
    self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
               (255, 0, 0), (0, 0, 255), (255, 0, 255),
               (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
               (255, 0, 0), (0, 0, 255), (255, 0, 255),
               (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
    self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255),
                      (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                      (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                      (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                      (255, 0, 0), (0, 0, 255)]

  def draw_pose(self, img, kps):
    """Draw the pose like https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/debugger.py#L191
    Arguments
      img: uint8 BGR
      kps: (17, 2) keypoint [[x, y]] coordinates
    """
    kps = np.array(kps, dtype=np.int32).reshape(self.num_joints, 2)
    for j in range(self.num_joints):
      cv2.circle(img, (kps[j, 0], kps[j, 1]), 3, self.colors_hp[j], -1)
    for j, e in enumerate(self.edges):
      if kps[e].min() > 0:
        cv2.line(img, (kps[e[0], 0], kps[e[0], 1]), (kps[e[1], 0], kps[e[1], 1]), self.ec[j], 2,
                 lineType=cv2.LINE_AA)
    return img

  def draw_box(self, img, x1, y1, x2, y2, cl):
    cl = int(cl)
    x1, y1, x2, y2 = int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))
    h = img.shape[0]
    width = max(1, int(h * 0.006))
    name = self.coco_names[cl].split()[-1]
    bgr_color = get_rgb_color(cl, len(self.coco_names))[::-1]
    # bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, width)
    # font background
    font_width = len(name) * self.char_width
    cv2.rectangle(img, (x1 - math.ceil(width / 2), y1 - self.font_size), (x1 + font_width, y1), bgr_color, -1)
    # text
    pil_img = Image.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil_img)
    draw.text((x1 + width, y1 - self.font_size), name, font=self.font, fill=(0, 0, 0, 255))
    img = np.array(pil_img)[..., ::-1].copy()
    return img
