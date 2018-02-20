#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import argparse
import os

from caffe.proto import caffe_pb2
from google.protobuf import text_format  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', action='store', dest='MODEL')
  params = parser.parse_args()

  net = caffe_pb2.NetParameter()
  with open(params.MODEL, 'r') as f:
    text_format.Merge(f.read(), net)

  # Remove dummy data layers
  data_layers = []
  for layer in net.layer:
    if layer.name == 'data':
      data_layers.append(layer)
  for layer in data_layers:
    net.layer.remove(layer)

  # Prepare standart input
  net.input = 'data'
  net.input_shape.append(-1)
  net.input_shape.append(3)
  net.input_shape.append(int(os.getenv('MODEL_IMAGE_SIZE')))
  net.input_shape.append(int(os.getenv('MODEL_IMAGE_SIZE')))

  # Serialize prototxt
  txt = text_format.MessageToString(net)

  # We cant insert strings into integer field using caffe_pb2 
  # as it's type safe, so do it with plain string replacement
  txt = txt.replace('dim: -1', 'dim: $#batch_size#$')

  with open(params.MODEL+'_1', 'w') as f:
    f.write(txt)
