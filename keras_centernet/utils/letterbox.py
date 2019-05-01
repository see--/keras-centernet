import cv2
import numpy as np


def letterbox_image(image, output_height, output_width):
  # https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html
  height, width = image.shape[:2]
  height_scale, width_scale = output_height / height, output_width / width
  scale = min(height_scale, width_scale)
  resize_height, resize_width = round(height * scale), round(width * scale)
  pad_top = (output_height - resize_height) // 2
  pad_left = (output_width - resize_width) // 2
  A = np.float32([[scale, 0.0], [0.0, scale]])
  B = np.float32([[pad_left], [pad_top]])
  M = np.hstack([A, B])
  # https://answers.opencv.org/question/33516/cv2warpaffine-results-in-an-image-shifted-by-05-pixel
  # This is different from `cv2.resize(image, (resize_width, resize_height))` & pad
  letterbox = cv2.warpAffine(image, M, (output_width, output_height))
  return letterbox, M


def invert_transform(M):
  # T = A @ x + B => x = A_inv @ (T - B) = A_inv @ T + (-A_inv @ B)
  A_inv = np.float32([[1. / M[0, 0], 0.0], [0.0, 1. / M[1, 1]]])
  B_inv = -A_inv @ M[:, 2:3]
  M_inv = np.hstack([A_inv, B_inv])
  return M_inv


def affine_transform_coords(coords, M):
  A, B = M[:2, :2], M[:2, 2:3]
  transformed_coords = A @ coords + B
  return transformed_coords


class LetterboxTransformer:
  def __init__(self, height, width):
    self.height = height
    self.width = width
    self.M = None
    self.M_inv = None

  def __call__(self, image):
    letterbox, M = letterbox_image(image, self.height, self.width)
    self.M = M
    self.M_inv = invert_transform(M)
    return letterbox

  def correct_box(self, x1, y1, x2, y2):
    coords = np.float32([[x1, x2], [y1, y2]])
    coords = affine_transform_coords(coords, self.M_inv)
    x1, y1, x2, y2 = coords[0, 0], coords[1, 0], coords[0, 1], coords[1, 1]
    return x1, y1, x2, y2
