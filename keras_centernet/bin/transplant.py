#!/usr/bin/env python3
from collections import defaultdict
from keras.layers import BatchNormalization, Conv2D, Activation
from keras_centernet.models.networks.hourglass import HourglassNetwork
import sys
import os
import argparse
import torch as th
sys.path.append(os.path.expanduser('~/Pytorch/CenterNet/src/lib/models'))
from model import load_model  # noqa
from networks.large_hourglass import HourglassNet  # noqa


def get_pymodel(heads, weight_path):
  pymodel = HourglassNet(heads, 2)
  pymodel = load_model(pymodel, weight_path)
  print("PyTorch Weights loaded")
  return pymodel


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, default='hpdet')
  parser.add_argument('--set-weights', action='store_true')
  args, _ = parser.parse_known_args()

  if not args.set_weights:
    print("Dry run for: %s" % args.task)
  else:
    print("Setting weights for: %s" % args.task)

  if args.task == 'ctdet':
    pytorch_weight_path = os.path.expanduser('~/Pytorch/CenterNet/models/ctdet_coco_hg.pth')
  else:
    pytorch_weight_path = os.path.expanduser('~/Pytorch/CenterNet/models/hpdet_coco_hg.pth')
  keras_weight_path = '%s_coco_hg.hdf5' % args.task

  kwargs = {
    'num_stacks': 2,
    'cnv_dim': 256,
    'inres': (512, 512),
    'weights': None,
  }
  if args.task == 'ctdet':
    heads = {
      'hm': 80,
      'reg': 2,
      'wh': 2
    }
  elif args.task == 'hpdet':
    heads = {
      'hm': 1,
      'hm_hp': 17,
      'hp_offset': 2,
      'hps': 34,
      'reg': 2,
      'wh': 2,
    }
  model = HourglassNetwork(heads=heads, **kwargs)
  print("Keras model loaded")
  # count layers
  num_conv_keras = sum(1 for l in model.layers if isinstance(l, Conv2D))
  num_bn_keras = sum(1 for l in model.layers if isinstance(l, BatchNormalization))
  num_relu_keras = sum(1 for l in model.layers if isinstance(l, Activation))

  pymodel = get_pymodel(heads, pytorch_weight_path)
  print("PyTorch model loaded")
  num_conv_pytorch = 0
  num_bn_pytorch = 0
  num_relu_pytorch = 0
  stack = list(pymodel.children())
  while len(stack) > 0:
    current = stack.pop(0)
    if isinstance(current, th.nn.Conv2d):
      num_conv_pytorch += 1
    elif isinstance(current, th.nn.BatchNorm2d):
      num_bn_pytorch += 1
    elif isinstance(current, th.nn.ReLU):
      num_relu_pytorch += 1
    else:
      stack.extend(current.children())

  print(num_conv_keras, num_conv_pytorch)
  print(num_bn_keras, num_bn_pytorch)
  print(num_relu_keras, num_relu_pytorch)

  state_dict = pymodel.state_dict()
  py_weights = []
  all_names = []
  base_names = []
  shape_mapping = defaultdict(list)
  for name, param in state_dict.items():
    if 'num_batches_tracked' in name:
      continue
    print(name, param.shape)
    # Pytorch: (out_dim, in_dim, k1, k2)
    # Keras: (k1, k2, in_dim, out_dim)
    py_shape = tuple(param.shape)
    if len(py_shape) == 4:
      py_shape = (py_shape[2], py_shape[3], py_shape[1], py_shape[0])
    py_weights.append([name, py_shape])
    shape_mapping[py_shape].append(name)
    base_names.append('.'.join(name.split('.')[:-1]))
    all_names.append(name)

  all_names = set(all_names)
  base_names = set(base_names)
  layers_with_weights = []
  matched_names = []
  unmatched_names = []
  py_matched = []
  num_matched_by_name = 0
  num_matched_by_shape = 0
  for layer in model.layers:
    if isinstance(layer, Conv2D) or isinstance(layer, BatchNormalization):
      weights = layer.get_weights()
    else:
      continue
    layer_name = layer.name

    layers_with_weights.append(layer_name)
    set_weights = []
    layer_names = []
    shapes_matched = True
    if layer_name in base_names:
      num_matched_by_name += 1
      if len(weights) == 1:
        tmp_name = layer_name + '.weight'
        py_weights = [pw for pw in py_weights if pw[0] != tmp_name]
        layer_names.append(tmp_name)
        py_weight = state_dict[tmp_name].numpy()
        if len(py_weight.shape) == 4:
          py_weight = py_weight.transpose(2, 3, 1, 0)
        set_weights.append(py_weight)
      elif len(weights) == 2:
        for s in ['.weight', '.bias']:
          tmp_name = layer_name + s
          py_weights = [pw for pw in py_weights if pw[0] != tmp_name]
          layer_names.append(tmp_name)
          py_weight = state_dict[tmp_name].numpy()
          if len(py_weight.shape) == 4:
            py_weight = py_weight.transpose(2, 3, 1, 0)
          set_weights.append(py_weight)
      else:
        assert len(weights) == 4, "Assumed BN"
        for s in ['.weight', '.bias', '.running_mean', '.running_var']:
          tmp_name = layer_name + s
          py_weights = [pw for pw in py_weights if pw[0] != tmp_name]
          layer_names.append(tmp_name)
          py_weight = state_dict[tmp_name].numpy()
          if len(py_weight.shape) == 4:
            py_weight = py_weight.transpose(2, 3, 1, 0)
          set_weights.append(py_weight)
    else:
      raise RuntimeError("Layer name not present: %s" % layer_name)
      print("missing: ", layer_name)

    if shapes_matched:
      layer_names_ = []
      base = '.'.join(layer_names[0].split('.')[:-1])
      for mn in layer_names:
        if mn.startswith(base):
          layer_names_.append(mn.split('.')[-1])
        else:
          raise RuntimeError("Bad base")

      print("Found match: %-55s -> %-55s, %-55s" % (layer_name, base, layer_names_))
      matched_names.append(layer_name)
      if args.set_weights:
        layer.set_weights(set_weights)
    else:
      unmatched_names.append(layer_name)

  print("\n\nMatched %d, Unmatched %d, Total %d" % (len(matched_names), len(unmatched_names), len(layers_with_weights)))
  print("By name: %d, By shape: %d" % (num_matched_by_name, num_matched_by_shape))
  for unmatched_name in unmatched_names:
    unmatched_weights = model.get_layer(unmatched_name).get_weights()
    print(unmatched_name, len(unmatched_weights), unmatched_weights[0].shape)

  py_weight_names = [p[0] for p in py_weights]
  left_over = [n for n in py_weight_names if n not in py_matched and 'batches_tracked' not in n]
  print("left_over: ", len(left_over), left_over)

  model.save_weights(keras_weight_path)
  print("Wrote keras weights to %s" % keras_weight_path)


if __name__ == '__main__':
  main()
