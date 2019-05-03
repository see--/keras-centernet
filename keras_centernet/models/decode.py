from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras import backend as K
from keras.layers import Lambda
import tensorflow as tf


def _nms(heat, kernel=3):
  hmax = K.pool2d(heat, (kernel, kernel), padding='same', pool_mode='max')
  keep = K.cast(K.equal(hmax, heat), K.floatx())
  return heat * keep


def _ctdet_decode(hm, reg, wh, k=100, output_stride=4):
  hm = K.sigmoid(hm)
  hm = _nms(hm)
  hm_shape = K.shape(hm)
  reg_shape = K.shape(reg)
  wh_shape = K.shape(wh)
  batch, width, cat = hm_shape[0], hm_shape[2], hm_shape[3]

  hm_flat = K.reshape(hm, (batch, -1))
  reg_flat = K.reshape(reg, (reg_shape[0], -1, reg_shape[-1]))
  wh_flat = K.reshape(wh, (wh_shape[0], -1, wh_shape[-1]))

  def _process_sample(args):
    _hm, _reg, _wh = args
    _scores, _inds = tf.math.top_k(_hm, k=100, sorted=True)
    _classes = K.cast(_inds % cat, 'float32')
    _inds = K.cast(_inds / cat, 'int32')
    _xs = K.cast(_inds % width, 'float32')
    _ys = K.cast(K.cast(_inds / width, 'int32'), 'float32')
    _wh = K.gather(_wh, _inds)
    _reg = K.gather(_reg, _inds)

    _xs = _xs + _reg[..., 0]
    _ys = _ys + _reg[..., 1]

    _x1 = _xs - _wh[..., 0] / 2
    _y1 = _ys - _wh[..., 1] / 2
    _x2 = _xs + _wh[..., 0] / 2
    _y2 = _ys + _wh[..., 1] / 2

    _x1 = output_stride * _x1
    _y1 = output_stride * _y1
    _x2 = output_stride * _x2
    _y2 = output_stride * _y2

    _detection = K.stack([_x1, _y1, _x2, _y2, _scores, _classes], -1)
    return _detection

  detections = K.map_fn(_process_sample, [hm_flat, reg_flat, wh_flat], dtype=K.floatx())
  return detections


def CtDetDecode(model, hm_index=3, reg_index=4, wh_index=5, k=100, output_stride=4):
  def _decode(args):
    hm, reg, wh = args
    return _ctdet_decode(hm, reg, wh, k=k, output_stride=output_stride)
  output = Lambda(_decode)([model.outputs[hm_index], model.outputs[reg_index], model.outputs[wh_index]])
  model = Model(model.input, output)
  return model
