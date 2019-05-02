import pytest
import numpy as np
import pickle
import cv2
from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image


def load_input_gold():
  inp = cv2.imread('tests/data/gold.png')
  with open('tests/data/output.p', 'rb') as f:
    gold = pickle.load(f)
  return inp, gold['hm'], gold['reg'], gold['wh']


def test_hourglass_predict():
  img, hm_gold, reg_gold, wh_gold = load_input_gold()
  kwargs = {
    'num_stacks': 2,
    'cnv_dim': 256,
    'weights': 'coco',
    'inres': (256, 256),
  }
  heads = {
    'hm': 80,
    'reg': 2,
    'wh': 2
  }
  model = HourglassNetwork(heads=heads, **kwargs)
  pimg = np.expand_dims(normalize_image(img), 0)
  output = model.predict(pimg)
  hm, reg, wh = output[3:]
  from IPython import embed; embed()
  assert hm.shape == hm_gold.shape
  assert reg.shape == reg_gold.shape
  assert wh.shape == wh_gold.shape


if __name__ == '__main__':
  test_hourglass_predict()
  pytest.main([__file__])
