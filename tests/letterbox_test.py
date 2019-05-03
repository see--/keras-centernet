import pytest
import numpy as np
import cv2
from keras_centernet.utils import letterbox as lb


def load_input_gold():
  inp = cv2.imread('tests/data/input.png')
  gold = cv2.imread('tests/data/letterbox_gold.png')
  return inp, gold


def test_letterbox_shape_training():
  lt = lb.LetterboxTransformer(256, 256)
  inp, gold = load_input_gold()
  letterbox = lt(inp)
  assert gold.shape == letterbox.shape


def test_letterbox_data_training():
  lt = lb.LetterboxTransformer(256, 256)
  inp, gold = load_input_gold()
  letterbox = lt(inp)
  assert np.all(gold == letterbox)


def test_letterbox_shape_testing():
  lt = lb.LetterboxTransformer(mode='testing', max_stride=128)
  test_shapes = {
    (427, 640, 3): (512, 768, 3),
    (230, 352, 3): (256, 384, 3),
    (428, 640, 3): (512, 768, 3),
  }
  for in_shape, out_shape in test_shapes.items():
    inp = np.zeros(in_shape, dtype='float32')
    output = lt(inp)
    assert output.shape == out_shape, "Expected %s got %s" % (out_shape, output.shape)
    inp_recovered = cv2.warpAffine(output, lt.M_inv, in_shape[:2][::-1])
    assert inp_recovered.shape == in_shape, "Expected %s got %s" % (in_shape, inp_recovered.shape)


def test_invert_transform():
  M = np.float32([[1.2, 0.0, 32.0], [0.0, 0.8, -4.0]])
  assert np.all(M == lb.invert_transform(lb.invert_transform(M)))


def test_affine_transform():
  np.random.seed(32)
  M = np.float32([[1.2, 0.0, 32.0], [0.0, 0.8, -4.0]])
  M_inv = lb.invert_transform(M)
  coords = np.random.randn(2, 12) + 5.0 * 20.0
  coords_transformed = coords.copy()
  coords_transformed = lb.affine_transform_coords(coords_transformed, M)
  coords_transformed = lb.affine_transform_coords(coords_transformed, M_inv)
  assert np.allclose(coords, coords_transformed)


if __name__ == '__main__':
  pytest.main([__file__])
