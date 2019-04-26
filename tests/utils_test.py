import pytest
import numpy as np
import cv2
from keras_centernet import utils


def load_input_gold():
  inp = cv2.imread('tests/data/input.png')
  gold = cv2.imread('tests/data/gold.png')
  return inp, gold


def test_letterbox_shape():
  inp, gold = load_input_gold()
  letterbox, _ = utils.letterbox_image(inp, 256, 256)
  assert gold.shape == letterbox.shape


def test_letterbox_data():
  inp, gold = load_input_gold()
  letterbox, _ = utils.letterbox_image(inp, 256, 256)
  assert np.all(gold == letterbox)


def test_invert_transform():
  M = np.float32([[1.2, 0.0, 32.0], [0.0, 0.8, -4.0]])
  assert np.all(M == utils.invert_transform(utils.invert_transform(M)))


if __name__ == '__main__':
  pytest.main([__file__])
