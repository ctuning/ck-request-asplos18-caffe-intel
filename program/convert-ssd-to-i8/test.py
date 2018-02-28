#!/usr/bin/env python
#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import argparse
import utils

CUR_DIR = os.path.realpath('.')
PREPARED_INFO_FILE = 'prepared_info.json'
BATCH_SIZE = int(os.getenv('CK_BATCH_SIZE', 1))
CAFFE_BIN_DIR = os.getenv('CK_ENV_LIB_CAFFE_BIN')

if __name__ == '__main__':
  if not os.path.isfile(PREPARED_INFO_FILE):
    print('Prepared info file not found. Please, run "convert" conmmand at first.')
    exit(1)

  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', action='store', dest='MODE')
  params = parser.parse_args()

  info = utils.read_json(PREPARED_INFO_FILE)
  image_count = int(info['lmdb_image_count'])
  src_prototxt_file = info['test_prototxt_f32'] if params.MODE == 'F32' else info['test_prototxt_i8']
  dst_prototxt_file = os.path.join(CUR_DIR, 'tmp.prototxt')

  print('\nPreparing {} ...'.format(src_prototxt_file))
  utils.prepare_test_prototxt(
    src_file = src_prototxt_file,
    dst_file = dst_prototxt_file,
    lmdb_dir = info['lmdb_dir'],
    batch_size = BATCH_SIZE,
    label_map_file = info['label_map_file'],
    name_size_file = info['name_size_file'],
    image_count = image_count
  )

  print('\nTest {} model...'.format(params.MODE))
  cmd = []
  cmd.append(os.path.join(CAFFE_BIN_DIR, 'caffe'))
  cmd.append('test')
  cmd.append('--model=' + dst_prototxt_file)
  cmd.append('--weights=' + info['weights'])
  cmd.append('--iterations=' + str(image_count/BATCH_SIZE))
  cmd.append('--detection')
  utils.run_command(cmd, 'test_'+params.MODE+'.log')

