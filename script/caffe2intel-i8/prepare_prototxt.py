#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import argparse
import os

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', action='store', dest='MODEL')
  parser.add_argument('-t', '--target', action='store', dest='TARGET')
  params = parser.parse_args()

  print('Prepare {} ...'.format(params.MODEL))

  target = ''

  input_layer_file = params.MODEL + '.input'
  if os.path.isfile(input_layer_file):
    with open(input_layer_file) as f:
      target = f.read()
      target = target.replace('$#val_mean#$', os.getenv('CK_ENV_MODEL_CAFFE_MEAN_BIN'))
      target = target.replace('$#val_lmdb#$', os.getenv('CK_CAFFE_IMAGENET_VAL_LMDB'))
      target = target.replace('$#val_batch_size#$', os.getenv('TRAIN_VAL_BATCH_SIZE'))
      target = target.replace('$#batch_size#$', os.getenv('DEPLOY_BATCH_SIZE'))

  with open(params.MODEL, 'r') as f:
    if target:
      target += '\n'
    target += f.read()

  with open(params.TARGET, 'w') as f:
    f.write(target)
