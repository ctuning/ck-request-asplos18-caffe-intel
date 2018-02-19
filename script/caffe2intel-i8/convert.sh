#!/bin/bash

#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

INPUT_TRAIN_VAL_PROTOTXT=${ORIGINAL_PACKAGE_DIR}/train_val.prototxt
INPUT_DEPLOY_PROTOTXT=${ORIGINAL_PACKAGE_DIR}/deploy.prototxt
ITERATIONS=64

echo "DEPLOY_BLOB_NAME=${DEPLOY_BLOB_NAME}"
echo "TRAIN_VAL_BLOB_NAME=${TRAIN_VAL_BLOB_NAME}"

function ConvertPrototxt() {
  INPUT_PROTOTXT=$1
  BLOB_NAME=$2
  echo
  echo "Convertion ${INPUT_PROTOTXT} ..."
  python ${CK_ENV_LIB_CAFFE}/../src/scripts/calibrator.py \
    --root=${CK_ENV_LIB_CAFFE} \
    --weights=${CK_ENV_MODEL_CAFFE_WEIGHTS} \
    --model=${INPUT_PROTOTXT} \
    --iterations=${ITERATIONS} \
    --blob_name="${BLOB_NAME}"
  if [ "${?}" != "0" ]
  then
    echo "Error: Convertion failed!"
    exit 1
  fi
}

if [ -f ${INPUT_TRAIN_VAL_PROTOTXT} ]; then
  ConvertPrototxt ${INPUT_TRAIN_VAL_PROTOTXT} ${TRAIN_VAL_BLOB_NAME}
  cp ${INPUT_TRAIN_VAL_PROTOTXT} ${INSTALL_DIR}
fi

if [ -f ${INPUT_DEPLOY_PROTOTXT} ]; then  
  #ConvertPrototxt ${INPUT_DEPLOY_PROTOTXT} ${DEPLOY_BLOB_NAME}
  cp ${INPUT_DEPLOY_PROTOTXT} ${INSTALL_DIR}
fi

cp ${CK_ENV_MODEL_CAFFE}/${CK_ENV_MODEL_CAFFE_MEAN_BIN_FILE} ${INSTALL_DIR}
#cp ${CK_ENV_MODEL_CAFFE}/${CK_ENV_MODEL_CAFFE_WEIGHTS_FILE} ${INSTALL_DIR}

exit 0
