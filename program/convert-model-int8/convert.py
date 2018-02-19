#
# Copyright (c) 2017 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os

CAFFE_INSTALL_DIR = os.getenv('CAFFE_INSTALL_DIR')
CK_ENV_MODEL_CAFFE_WEIGHTS = os.getenv('CK_ENV_MODEL_CAFFE_WEIGHTS')
CK_CAFFE_ITERATIONS = os.getenv('CK_CAFFE_ITERATIONS')
MODEL_PROTOTXT = 'train_val.prototxt'
SCRIPT_FILE = os.path.join(CAFFE_INSTALL_DIR, '..', 'src', 'scripts', 'calibrator.py')
BLOB_NAME = ''

print('CAFFE_INSTALL_DIR={}'.format(CAFFE_INSTALL_DIR))
print('CK_ENV_MODEL_CAFFE_WEIGHTS={}'.format(CK_ENV_MODEL_CAFFE_WEIGHTS))
print('CK_CAFFE_ITERATIONS={}'.format(CK_CAFFE_ITERATIONS))

#os.system(SCRIPT_FILE \
#    + ' --root=' + CAFFE_INSTALL_DIR \
#    + ' --model=' + MODEL_PROTOTXT \
#    + ' --weights=' + CK_ENV_MODEL_CAFFE_WEIGHTS \
#    + ' --iterations=' + CK_CAFFE_ITERATIONS \
#    + ' --blob_name=' + BLOB_NAME
#  )
  
  
  
  
  
   
