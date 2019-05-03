import pytest
import numpy as np
import cv2
from keras_centernet.utils import letterbox as lb
np.random.seed(32)


def load_input_gold():
  inp = cv2.imread('tests/data/input.png')
  gold = cv2.imread('tests/data/letterbox_gold.png')
  return inp, gold


def test_letterbox_shape():
  inp, gold = load_input_gold()
  letterbox, _ = lb.letterbox_image(inp, 256, 256)
  assert gold.shape == letterbox.shape


def test_letterbox_data():
  inp, gold = load_input_gold()
  letterbox, _ = lb.letterbox_image(inp, 256, 256)
  assert np.all(gold == letterbox)


def test_invert_transform():
  M = np.float32([[1.2, 0.0, 32.0], [0.0, 0.8, -4.0]])
  assert np.all(M == lb.invert_transform(lb.invert_transform(M)))


def test_affine_transform():
  M = np.float32([[1.2, 0.0, 32.0], [0.0, 0.8, -4.0]])
  M_inv = lb.invert_transform(M)
  coords = np.random.randn(2, 12) + 5.0 * 20.0
  coords_transformed = coords.copy()
  coords_transformed = lb.affine_transform_coords(coords_transformed, M)
  coords_transformed = lb.affine_transform_coords(coords_transformed, M_inv)
  assert np.allclose(coords, coords_transformed)


if __name__ == '__main__':
  pytest.main([__file__])
