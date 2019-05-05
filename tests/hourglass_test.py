import pytest
import numpy as np
import pickle
import cv2
from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
import sys
sys.path.append('/home/steffen/Pytorch/CenterNet/src')
from lib.models.networks.large_hourglass import HourglassNet
from lib.models.model import load_model
import torch as th


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
    'weights': 'ctdet_coco',
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


def test_hourglass_output():
  kwargs = {
    'num_stacks': 2,
    'cnv_dim': 256,
    'weights': 'ctdet_coco',
    'inres': (256, 256),
  }
  heads = {
    'hm': 80,
    'reg': 2,
    'wh': 2
  }
  py_model = HourglassNet(heads)
  state_dict = th.load('/home/steffen/Pytorch/CenterNet/src/hg_weights.pth')
  py_model.load_state_dict(state_dict)
  py_model = load_model(py_model, "/home/steffen/Pytorch/CenterNet/models/ctdet_coco_hg.pth")
  py_model.eval()
  device = 'cpu'
  py_model = py_model.to(device)
  img = cv2.imread('tests/data/letterbox_gold.png')
  pimg = np.float32(img) / 127.5 - 1.0
  pimg = np.expand_dims(pimg, 0)
  py_pimg = th.from_numpy(pimg.transpose(0, 3, 1, 2))
  py_pimg = py_pimg.to(device)
  py_output = py_model(py_pimg)[-1]
  py_hm = py_output['hm'].detach().cpu().numpy().transpose(0, 2, 3, 1)
  py_reg = py_output['reg'].detach().cpu().numpy().transpose(0, 2, 3, 1)
  py_wh = py_output['wh'].detach().cpu().numpy().transpose(0, 2, 3, 1)

  model = HourglassNetwork(heads=heads, **kwargs)
  output = model.predict(pimg)
  hm = output[3]
  reg = output[4]
  wh = output[5]

  print("HM: ", py_hm.mean().round(3), py_hm.std().round(3).round(3), py_hm.min().round(3), py_hm.max().round(3), " | ", hm.mean().round(3), hm.std().round(3).round(3), hm.min().round(3), hm.max().round(3))  # noqa
  print("REG: ", py_reg.mean().round(3), py_reg.std().round(3).round(3), py_reg.min().round(3), py_reg.max().round(3), " | ", reg.mean().round(3), reg.std().round(3).round(3), reg.min().round(3), reg.max().round(3))  # noqa
  print("WH: ", py_wh.mean().round(3), py_wh.std().round(3).round(3), py_wh.min().round(3), py_wh.max().round(3), " | ", wh.mean().round(3), wh.std().round(3).round(3), wh.min().round(3), wh.max().round(3))  # noqa

  from IPython import embed; embed()


if __name__ == '__main__':
  test_hourglass_output()
  pytest.main([__file__])
