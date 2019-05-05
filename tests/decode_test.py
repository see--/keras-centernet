import pytest
import numpy as np
from keras_centernet.models.decode import _ctdet_decode, _hpdet_decode
from keras import backend as K
import os
import pickle


def test_ctdet_decode():
  np.random.seed(32)
  hm = np.random.randn(2, 64, 64, 80)
  reg = np.random.randn(2, 64, 64, 2) * 10.0
  wh = np.random.randn(2, 64, 64, 2) * 20.0

  keras_hm = K.constant(hm)
  keras_reg = K.constant(reg)
  keras_wh = K.constant(wh)

  keras_detections = K.eval(_ctdet_decode(keras_hm, keras_reg, keras_wh, output_stride=1))

  gold_fn = 'tests/data/ctdet_decode_gold.p'
  if not os.path.exists(gold_fn):
    import torch as th
    import sys
    sys.path.append(os.path.expanduser('~/Pytorch/CenterNet/src'))
    from lib.models.decode import ctdet_decode  # noqa
    py_hm = th.from_numpy(hm.transpose(0, 3, 1, 2)).float()
    py_hm.sigmoid_()
    py_reg = th.from_numpy(reg.transpose(0, 3, 1, 2)).float()
    py_wh = th.from_numpy(wh.transpose(0, 3, 1, 2)).float()
    py_detections = ctdet_decode(py_hm, py_reg, py_wh).detach().numpy()
    with open(gold_fn, 'wb') as f:
      pickle.dump(py_detections, f)
  else:
    with open(gold_fn, 'rb') as f:
      py_detections = pickle.load(f)
  assert np.allclose(keras_detections, py_detections)


def test_hpdet_decode():
  np.random.seed(32)
  hm = np.random.randn(2, 64, 64, 1)
  hm_hp = np.random.randn(2, 64, 64, 17)
  hp_offset = np.random.randn(2, 64, 64, 2) * 10.0
  kps = np.random.randn(2, 64, 64, 34) * 5.0
  reg = np.random.randn(2, 64, 64, 2) * 10.0
  wh = np.random.randn(2, 64, 64, 2) * 20.0

  keras_hm = K.constant(hm)
  keras_hm_hp = K.constant(hm_hp)
  keras_hp_offset = K.constant(hp_offset)
  keras_kps = K.constant(kps)
  keras_reg = K.constant(reg)
  keras_wh = K.constant(wh)

  keras_detections = K.eval(_hpdet_decode(
    keras_hm, keras_wh, keras_kps, keras_reg, keras_hm_hp, keras_hp_offset, output_stride=1))
  gold_fn = 'tests/data/hpdet_decode_gold.p'
  if not os.path.exists(gold_fn):
    import torch as th
    import sys
    sys.path.append(os.path.expanduser('~/Pytorch/CenterNet/src'))
    from lib.models.decode import multi_pose_decode as hpdet_decode  # noqa
    py_hm = th.from_numpy(hm.transpose(0, 3, 1, 2)).float()
    py_hm.sigmoid_()
    py_hm_hp = th.from_numpy(hm_hp.transpose(0, 3, 1, 2)).float()
    py_hm_hp.sigmoid_()
    py_kps = th.from_numpy(kps.transpose(0, 3, 1, 2)).float()
    py_reg = th.from_numpy(reg.transpose(0, 3, 1, 2)).float()
    py_wh = th.from_numpy(wh.transpose(0, 3, 1, 2)).float()
    py_hp_offset = th.from_numpy(hp_offset.transpose(0, 3, 1, 2)).float()
    py_detections = hpdet_decode(py_hm, py_wh, py_kps, py_reg, py_hm_hp, py_hp_offset).detach().numpy()
    py_kps = py_detections[..., 5:-1].copy()
    py_kps_x = py_kps[..., ::2]
    py_kps_y = py_kps[..., 1::2]
    py_kps = np.concatenate([py_kps_x, py_kps_y], 2)
    py_detections[..., 5:-1] = py_kps
    with open(gold_fn, 'wb') as f:
      pickle.dump(py_detections, f)
  else:
    with open(gold_fn, 'rb') as f:
      py_detections = pickle.load(f)

  assert np.allclose(keras_detections, py_detections)


if __name__ == '__main__':
  pytest.main([__file__])
