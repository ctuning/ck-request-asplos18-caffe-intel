#!/usr/bin/env python
#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import uuid
import numpy as np
from google.protobuf import text_format
import utils

CUR_DIR = os.path.realpath('.')
MODEL_DIR = os.getenv('CK_ENV_MODEL_CAFFE')
IMAGES_DIR = os.getenv('CK_ENV_DATASET_IMAGE_DIR')
LABELS_DIR = os.getenv('CK_ENV_DATASET_LABELS_DIR')
LABEL_MAP_FILE = os.getenv('CK_ENV_MODEL_CAFFE_LABELMAP')
IMAGES_PERCENT = float(os.getenv('CK_IMAGES_PERCENT'))
IMAGE_LIST_FILE = os.path.join(CUR_DIR, 'image_list.txt')
NAME_SIZE_FILE = os.path.join(CUR_DIR, 'names_size.txt')
TARGET_IMG_W = utils.model_img_w(MODEL_DIR)
TARGET_IMG_H = utils.model_img_h(MODEL_DIR)
LMDB_TARGET_DIR = os.path.join(CUR_DIR, 'lmdb')
LMDB_IMAGE_COUNT = 0
CAFFE_DIR = os.getenv('CK_ENV_LIB_CAFFE')
CAFFE_BIN_DIR = os.getenv('CK_ENV_LIB_CAFFE_BIN')
SRC_TEST_PROTOTXT_FILE = os.path.join(MODEL_DIR, 'test.prototxt')
TMP_TEST_PROTOTXT_FILE = os.path.join(CUR_DIR, 'test.prototxt')
DST_TEST_PROTOTXT_FILE = os.path.join(CUR_DIR, 'test_quantized.prototxt')
SRC_DEPLOY_PROTOTXT_FILE = os.path.join(MODEL_DIR, 'deploy.prototxt')
TMP_DEPLOY_PROTOTXT_FILE = os.path.join(CUR_DIR, 'deploy.prototxt')
DST_DEPLOY_PROTOTXT_FILE = os.path.join(CUR_DIR, 'deploy_quantized.prototxt')
BATCH_SIZE = int(os.getenv('CK_BATCH_SIZE', 1))
WEIGHTS_FILE = os.getenv('CK_ENV_MODEL_CAFFE_WEIGHTS')
PYTHON = os.getenv('CK_ENV_COMPILER_PYTHON_FILE')
PREPARED_INFO_FILE = 'prepared_info.json'

########################################################################

def make_image_list():
  '''
  Makes list of image files and their annotation files.
  '''
  global LMDB_IMAGE_COUNT
  all_images = os.listdir(IMAGES_DIR)
  all_images = np.random.permutation(all_images) 
  LMDB_IMAGE_COUNT = int(len(all_images) * IMAGES_PERCENT / 100.0)
  print('Total images count: {}'.format(len(all_images)))
  print('Test images count: {}'.format(LMDB_IMAGE_COUNT))

  with open(IMAGE_LIST_FILE, 'w') as f:
    for image_file in all_images[:LMDB_IMAGE_COUNT]:
      image_path = os.path.join(IMAGES_DIR, image_file)
      label_file = image_file[:-3] + 'txt'
      label_path = os.path.join(LABELS_DIR, label_file)
      f.write(image_path + ' ' + label_path + '\n')

  # Generate image name and size infomation.
  cmd = []
  cmd.append(os.path.join(CAFFE_BIN_DIR, 'get_image_size'))
  cmd.append('""') # we can leave root path empty as our file list contains absolute pathes
  cmd.append(IMAGE_LIST_FILE)
  cmd.append(NAME_SIZE_FILE)
  utils.run_command(cmd, 'get_image_size.log')

########################################################################

def make_lmdb():
  '''
  Use convert_annoset tool.
  The tool takes a list file as parameter and writes
  all listed images and its labels into LBDM database.
  '''
  utils.rmdir(LMDB_TARGET_DIR)

  cmd = []
  cmd.append(os.path.join(CAFFE_BIN_DIR, 'convert_annoset'))
  cmd.append('--anno_type=detection')
  cmd.append('--label_type=txt')
  cmd.append('--label_map_file=' + LABEL_MAP_FILE)
  cmd.append('--resize_height=' + str(TARGET_IMG_H))
  cmd.append('--resize_width=' + str(TARGET_IMG_W))
  cmd.append('--backend=lmdb')
  cmd.append('--encoded')
  cmd.append('--encode_type=jpg')
  cmd.append('""') # we can leave root path empty as our file list contains absolute pathes
  cmd.append(IMAGE_LIST_FILE)
  cmd.append(LMDB_TARGET_DIR)
  utils.run_command(cmd, 'convert_annoset.log')

########################################################################

def convert_prototxt():
  '''
  Converts f32 prototxt into quantized version using Intel's tool calibrator.py
  '''
  #LMDB_IMAGE_COUNT=17125
  cmd = []
  cmd.append(PYTHON)
  cmd.append(os.path.join(CAFFE_DIR, '..', 'src', 'scripts', 'calibrator.py'))
  cmd.append('--root=' + CAFFE_DIR)
  cmd.append('--weights=' + WEIGHTS_FILE)
  cmd.append('--model=' + TMP_TEST_PROTOTXT_FILE)
  cmd.append('--iterations=' + str(LMDB_IMAGE_COUNT / BATCH_SIZE))
  cmd.append('--blob_name=detection_eval')
  utils.run_command(cmd, 'convert_prototxt.log')


########################################################################

def postprocess_test_prototxt():
  '''
  Prepares final version of quantized prototxt
  replacing real paths and batch size to respective ck-variables.
  '''
  net = utils.read_prototxt(DST_TEST_PROTOTXT_FILE)

  for layer in net.layer:
    if layer.name == 'data':
      layer.data_param.source = '$#val_lmdb#$'
      layer.data_param.batch_size = 0
      layer.annotated_data_param.label_map_file = '$#path_to_labelmap#$'
    elif layer.name == 'detection_out':
      p = layer.detection_output_param.save_output_param
      p.label_map_file = '$#path_to_labelmap#$'
      p.name_size_file = '$#path_to_name_size#$'
    elif layer.name == 'detection_eval':
      layer.detection_evaluate_param.name_size_file = '$#path_to_name_size#$'

  # We cant insert strings into integer field using caffe_pb2 
  # as it's type safe, so do it with plain string replacement
  txt = text_format.MessageToString(net)
  txt = txt.replace('batch_size: 0', 'batch_size: $#val_batch_size#$')
  txt = txt.replace('num_test_image: 0', 'num_test_image: $#num_test_image#$')

  utils.write_text(DST_TEST_PROTOTXT_FILE, txt)

########################################################################

def make_deploy_prototxt():
  '''
  Makes deploy.prototxt model based on test.prototxt.
  The differences between them are first and last layers.
  '''
  net = utils.read_prototxt(DST_TEST_PROTOTXT_FILE)

  # Remove first and last layers
  layers_to_remove = []
  for layer in net.layer:
    if layer.name == 'data' or layer.name == 'detection_eval':
      layers_to_remove.append(layer)
  for layer in layers_to_remove:
    net.layer.remove(layer)

  # Prepare standart input
  net.input.append('data')
  shape = net.input_shape.add()
  shape.dim.append(0)
  shape.dim.append(3)
  shape.dim.append(TARGET_IMG_H)
  shape.dim.append(TARGET_IMG_W)

  # Not sure why there params should be removed,
  # but they are not presented in deploy.prototxt from packages
  # caffemodel-ssd-coco-* and caffemodel-ssd-voc-*
  for layer in net.layer:
    if layer.name == 'detection_out':
      p = layer.detection_output_param.save_output_param
      p.label_map_file = '$#path_to_labelmap#$'
      p.output_name_prefix = ''
      p.output_format = ''
      p.name_size_file = ''
      p.num_test_image = 0

  # We can't insert strings into integer field using caffe_pb2 
  # as it's type safe, so do it with plain string replacement
  txt = text_format.MessageToString(net)
  txt = txt.replace('dim: 0', 'dim: $#batch_size#$', 1)

  utils.write_text(DST_DEPLOY_PROTOTXT_FILE, txt)

########################################################################

if __name__ == '__main__':
  print('Model dir: {}'.format(MODEL_DIR))
  print('Images dir: {}'.format(IMAGES_DIR))
  print('Labels dir: {}'.format(LABELS_DIR))
  print('Images percent: {}'.format(IMAGES_PERCENT))
  print('Target image size (HxW): {}x{}'.format(TARGET_IMG_H, TARGET_IMG_W))

  print('\nInitializing...')
  # Path ${CK-TOOLS}/lib-caffe-*/install/python/caffe/ is not a package
  # but it should to be as calibrator.py need to import caffe_pb2.py
  caffe_package_init_file = os.path.join(CAFFE_DIR, 'python', 'caffe', '__init__.py')
  caffe_package_init_file_uid = None
  if not os.path.isfile(caffe_package_init_file):
    caffe_package_init_file_uid = '#' + str(uuid.uuid1())
    print('Making {} ...'.format(caffe_package_init_file))
    print(caffe_package_init_file_uid)
    with open(caffe_package_init_file, 'w') as f:
      f.write(caffe_package_init_file_uid)
  else:
    print('{} already exists'.format(caffe_package_init_file))

  if os.path.isfile(PREPARED_INFO_FILE):
    os.remove(PREPARED_INFO_FILE)

  try:
    print('\nMaking image list files...')
    make_image_list()

    print('\nMaking lmdb database...')
    make_lmdb()

    print('\nPreparing {} ...'.format(SRC_TEST_PROTOTXT_FILE))
    utils.prepare_test_prototxt(
      src_file = SRC_TEST_PROTOTXT_FILE,
      dst_file = TMP_TEST_PROTOTXT_FILE,
      lmdb_dir = LMDB_TARGET_DIR,
      batch_size = BATCH_SIZE,
      label_map_file = LABEL_MAP_FILE,
      name_size_file = NAME_SIZE_FILE,
      image_count = LMDB_IMAGE_COUNT
    )

    print('\nConverting {} ...'.format(TMP_TEST_PROTOTXT_FILE))
    convert_prototxt()

    print('\Making {} ...'.format(DST_DEPLOY_PROTOTXT_FILE))
    make_deploy_prototxt()

    print('\nPostprocessing {} ...'.format(DST_TEST_PROTOTXT_FILE))
    postprocess_test_prototxt()

    info = {}
    info['weights'] = WEIGHTS_FILE
    info['lmdb_dir'] = LMDB_TARGET_DIR
    info['lmdb_image_count'] = LMDB_IMAGE_COUNT
    info['test_prototxt_f32'] = SRC_TEST_PROTOTXT_FILE
    info['test_prototxt_i8'] = DST_TEST_PROTOTXT_FILE
    info['deploy_prototxt_f32'] = SRC_DEPLOY_PROTOTXT_FILE
    info['deploy_prototxt_i8'] = DST_DEPLOY_PROTOTXT_FILE
    info['label_map_file'] = LABEL_MAP_FILE
    info['name_size_file'] = NAME_SIZE_FILE
    utils.write_json(PREPARED_INFO_FILE, info)

  finally:
    print('\nFinalizing...')
    if caffe_package_init_file_uid:
      if os.path.isfile(caffe_package_init_file):
        existed_uid = utils.read_text(caffe_package_init_file)
        print('{} exists'.format(caffe_package_init_file))
        if existed_uid == caffe_package_init_file_uid:
          os.remove(caffe_package_init_file)
          print('Removed')
