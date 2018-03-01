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

  print('Postprocessing {} ...'.format(params.MODEL))

  net = caffe_pb2.NetParameter()
  with open(params.MODEL, 'r') as f:
    text_format.Merge(f.read(), net)

  # Adjust input layer
  # NB: Currently, we assume that there is only one input layer.
  for layer in net.layer:
    if layer.name == 'data':
      layer.transform_param.mean_file = '$#val_mean#$'
      layer.data_param.source = '$#val_lmdb#$'
      layer.data_param.batch_size = 0

  # Serialize prototxt
  txt = text_format.MessageToString(net)

  # We cannot insert strings into integer field using caffe_pb2 
  # as it's type safe, so do it with plain string replacement
  txt = txt.replace('batch_size: 0', 'batch_size: $#val_batch_size#$')

  with open(params.MODEL, 'w') as f:
    f.write(txt)  

