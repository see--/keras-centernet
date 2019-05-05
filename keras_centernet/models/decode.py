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
    _scores, _inds = tf.math.top_k(_hm, k=k, sorted=True)
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

    # rescale to image coordinates
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
  output = Lambda(_decode)([model.outputs[i] for i in [hm_index, reg_index, wh_index]])
  model = Model(model.input, output)
  return model


def _hpdet_decode(hm, wh, kps, reg, hm_hp, hp_offset, k=100, output_stride=4):
  hm = K.sigmoid(hm)
  hm = _nms(hm)
  hm_shape = K.shape(hm)
  reg_shape = K.shape(reg)
  wh_shape = K.shape(wh)
  kps_shape = K.shape(kps)
  batch, width, cat = hm_shape[0], hm_shape[2], hm_shape[3]

  hm_flat = K.reshape(hm, (batch, -1))
  reg_flat = K.reshape(reg, (reg_shape[0], -1, reg_shape[-1]))
  wh_flat = K.reshape(wh, (wh_shape[0], -1, wh_shape[-1]))
  kps_flat = K.reshape(kps, (kps_shape[0], -1, kps_shape[-1]))

  hm_hp = K.sigmoid(hm_hp)
  hm_hp = _nms(hm_hp)
  hm_hp_shape = K.shape(hm_hp)
  hp_offset_shape = K.shape(hp_offset)

  hm_hp_flat = K.reshape(hm_hp, (hm_hp_shape[0], -1, hm_hp_shape[-1]))
  hp_offset_flat = K.reshape(hp_offset, (hp_offset_shape[0], -1, hp_offset_shape[-1]))

  def _process_sample(args):
    _hm, _reg, _wh, _kps, _hm_hp, _hp_offset = args
    _scores, _inds = tf.math.top_k(_hm, k=k, sorted=True)
    _classes = K.cast(_inds % cat, 'float32')
    _inds = K.cast(_inds / cat, 'int32')
    _xs = K.cast(_inds % width, 'float32')
    _ys = K.cast(K.cast(_inds / width, 'int32'), 'float32')
    _wh = K.gather(_wh, _inds)
    _reg = K.gather(_reg, _inds)
    _kps = K.gather(_kps, _inds)

    # shift keypoints by their center
    _kps_x = _kps[:, ::2]
    _kps_y = _kps[:, 1::2]
    _kps_x = _kps_x + K.expand_dims(_xs, -1)  # k x J
    _kps_y = _kps_y + K.expand_dims(_ys, -1)  # k x J
    _kps = K.stack([_kps_x, _kps_y], -1)  # k x J x 2

    _xs = _xs + _reg[..., 0]
    _ys = _ys + _reg[..., 1]

    _x1 = _xs - _wh[..., 0] / 2
    _y1 = _ys - _wh[..., 1] / 2
    _x2 = _xs + _wh[..., 0] / 2
    _y2 = _ys + _wh[..., 1] / 2

    # snap center keypoints to the closest heatmap keypoint
    def _process_channel(args):
      __kps, __hm_hp = args
      thresh = 0.1
      __hm_scores, __hm_inds = tf.math.top_k(__hm_hp, k=k, sorted=True)
      __hm_xs = K.cast(__hm_inds % width, 'float32')
      __hm_ys = K.cast(K.cast(__hm_inds / width, 'int32'), 'float32')
      __hp_offset = K.gather(_hp_offset, __hm_inds)
      __hm_xs = __hm_xs + __hp_offset[..., 0]
      __hm_ys = __hm_ys + __hp_offset[..., 1]
      mask = K.cast(__hm_scores > thresh, 'float32')
      __hm_scores = (1. - mask) * -1. + mask * __hm_scores
      __hm_xs = (1. - mask) * -10000. + mask * __hm_xs
      __hm_ys = (1. - mask) * -10000. + mask * __hm_ys
      __hm_kps = K.stack([__hm_xs, __hm_ys], -1)  # k x 2
      __broadcast_hm_kps = K.expand_dims(__hm_kps, 1)  # k x 1 x 2
      __broadcast_kps = K.expand_dims(__kps, 0)  # 1 x k x 2
      dist = K.sqrt(K.sum(K.pow(__broadcast_kps - __broadcast_hm_kps, 2), 2))  # k, k
      min_dist = K.min(dist, 0)
      min_ind = K.argmin(dist, 0)
      __hm_scores = K.gather(__hm_scores, min_ind)
      __hm_kps = K.gather(__hm_kps, min_ind)
      mask = (K.cast(__hm_kps[..., 0] < _x1, 'float32') + K.cast(__hm_kps[..., 0] > _x2, 'float32') +
              K.cast(__hm_kps[..., 1] < _y1, 'float32') + K.cast(__hm_kps[..., 1] > _y2, 'float32') +
              K.cast(__hm_scores < thresh, 'float32') +
              K.cast(min_dist > 0.3 * (K.maximum(_wh[..., 0], _wh[..., 1])), 'float32'))
      mask = K.expand_dims(mask, -1)
      mask = K.cast(mask > 0, 'float32')
      __kps = (1. - mask) * __hm_kps + mask * __kps
      return __kps

    _kps = K.permute_dimensions(_kps, (1, 0, 2))  # J x k x 2
    _hm_hp = K.permute_dimensions(_hm_hp, (1, 0))  # J x -1
    _kps = K.map_fn(_process_channel, [_kps, _hm_hp], dtype='float32')
    _kps = K.reshape(K.permute_dimensions(_kps, (1, 2, 0)), (k, -1))  # k x J * 2

    # rescale to image coordinates
    _x1 = output_stride * _x1
    _y1 = output_stride * _y1
    _x2 = output_stride * _x2
    _y2 = output_stride * _y2
    _kps = output_stride * _kps

    _boxes = K.stack([_x1, _y1, _x2, _y2], -1)
    _scores = K.expand_dims(_scores, -1)
    _classes = K.expand_dims(_classes, -1)
    _detection = K.concatenate([_boxes, _scores, _kps, _classes], -1)
    return _detection

  detections = K.map_fn(_process_sample,
                        [hm_flat, reg_flat, wh_flat, kps_flat, hm_hp_flat, hp_offset_flat], dtype='float32')
  return detections


def HpDetDecode(model, hm_index=6, wh_index=11, kps_index=9, reg_index=10, hm_hp_index=7, hp_offset_index=8,
                k=100, output_stride=4):
  def _decode(args):
    hm, wh, kps, reg, hm_hp, hp_offset = args
    return _hpdet_decode(hm, wh, kps, reg, hm_hp, hp_offset, k=k, output_stride=output_stride)

  output = Lambda(_decode)(
    [model.outputs[i] for i in [hm_index, wh_index, kps_index, reg_index, hm_hp_index, hp_offset_index]])
  model = Model(model.input, output)
  return model
