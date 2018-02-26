#!/usr/bin/env python
#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import numpy as np
import utils

MODEL_DIR = os.getenv('CK_ENV_MODEL_CAFFE')
IMAGES_DIR = os.getenv('CK_ENV_DATASET_IMAGE_DIR')
LABELS_DIR = os.getenv('CK_ENV_DATASET_LABELS_DIR')
LABEL_MAP_FILE = os.getenv('CK_ENV_MODEL_CAFFE_LABELMAP')
IMAGES_PERCENT = float(os.getenv('CK_IMAGES_PERCENT'))
IMAGE_LIST_FILE = 'image_list.txt'
TARGET_IMG_W = utils.model_img_w(MODEL_DIR)
TARGET_IMG_H = utils.model_img_h(MODEL_DIR)
LMDB_TARGET_DIR = 'lmdb'
CAFFE_BIN_DIR = os.getenv('CK_ENV_LIB_CAFFE_BIN')


def make_image_list():
  '''
  Makes list of image files and their annotation files.
  '''
  all_images = os.listdir(IMAGES_DIR)
  all_images = np.random.permutation(all_images) 
  img_count = int(len(all_images) * IMAGES_PERCENT / 100.0)
  print('Total images count: {}'.format(len(all_images)))
  print('Test images count: {}'.format(img_count))

  with open(IMAGE_LIST_FILE, 'w') as f:
    for image_file in all_images[:img_count]:
      image_path = os.path.join(IMAGES_DIR, image_file)
      label_file = image_file[:-3] + 'txt'
      label_path = os.path.join(LABELS_DIR, label_file)
      f.write(image_path + ' ' + label_path + '\n')
      

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
  cmd.append('') # we can leave root path empty as our file list contains absolute pathes
  cmd.append(IMAGE_LIST_FILE)
  cmd.append(LMDB_TARGET_DIR)
  utils.run_command(cmd)


if __name__ == '__main__':
  print('Model dir: {}'.format(MODEL_DIR))
  print('Images dir: {}'.format(IMAGES_DIR))
  print('Labels dir: {}'.format(LABELS_DIR))
  print('Images percent: {}'.format(IMAGES_PERCENT))
  print('Target image size (HxW): {}x{}'.format(TARGET_IMG_H, TARGET_IMG_W))

  print('\nMaking image list files...')
  make_image_list()

  print('\nMaking lmdb database...')
  make_lmdb()

