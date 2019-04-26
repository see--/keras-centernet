import math
import cv2
import numpy as np

colors = [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
coco_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',]  # noqa


def get_color(c, x, max_value):
  # https://github.com/pjreddie/darknet/blob/master/src/image.c
  global colors
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


def letterbox_image(image, output_height, output_width):
  height, width = image.shape[:2]
  height_scale, width_scale = output_height / height, output_width / width
  scale = min(height_scale, width_scale)
  resize_height, resize_width = round(height * scale), round(width * scale)
  pad_top = (output_height - resize_height) // 2
  pad_left = (output_width - resize_width) // 2
  S = np.float32([[scale, 0.0], [0.0, scale]])
  T = np.float32([[pad_left], [pad_top]])
  M = np.hstack([S, T])
  # https://answers.opencv.org/question/33516/cv2warpaffine-results-in-an-image-shifted-by-05-pixel
  # This is different from `cv2.resize(image, (resize_width, resize_height))` & pad
  letterbox = cv2.warpAffine(image, M, (output_width, output_height))
  return letterbox, M


def invert_transform(M):
  S = np.float32([[1. / M[0, 0], 0.0], [0.0, 1. / M[1, 1]]])
  T = -M[:, 2:3]
  M_inv = np.hstack([S, T])
  return M_inv
