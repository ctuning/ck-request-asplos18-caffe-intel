#!/bin/bash

#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

function ConvertPrototxt() {
  PROTOTXT=$1
  BLOB_NAME=$2
  ITERATIONS=$3
  INPUT_PROTOTXT=${PROTOTXT}.prototxt
  OUTPUT_PROTOTXT=${PROTOTXT}_quantized.prototxt
  THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

  echo
  echo "Convertion ${INPUT_PROTOTXT} ..."
  echo

  python ${THIS_SCRIPT_DIR}/prepare_prototxt.py \
    --model=${ORIGINAL_PACKAGE_DIR}/${INPUT_PROTOTXT} \
    --target=${INSTALL_DIR}/${INPUT_PROTOTXT}

  #python ${CK_ENV_LIB_CAFFE}/../src/scripts/calibrator.py \
  #  --root=${CK_ENV_LIB_CAFFE} \
  #  --weights=${CK_ENV_MODEL_CAFFE_WEIGHTS} \
  #  --model=${INSTALL_DIR}/${INPUT_PROTOTXT} \
  #  --iterations=${ITERATIONS}
  #  --blob_name=${BLOB_NAME}

  python ${THIS_SCRIPT_DIR}/finalize_prototxt.py \
    --model=${ORIGINAL_PACKAGE_DIR}/${INPUT_PROTOTXT} \
    --target=${INSTALL_DIR}/${OUTPUT_PROTOTXT}

  #rm ${INSTALL_DIR}/${INPUT_PROTOTXT}
}

ConvertPrototxt 'train_val' ${TRAIN_VAL_BLOB_NAME} ${TRAIN_VAL_ITERATIONS}
ConvertPrototxt 'deploy' ${DEPLOY_BLOB_NAME} ${DEPLOY_ITERATIONS}

#cp ${CK_ENV_MODEL_CAFFE}/${CK_ENV_MODEL_CAFFE_MEAN_BIN_FILE} ${INSTALL_DIR}
#cp ${CK_ENV_MODEL_CAFFE}/${CK_ENV_MODEL_CAFFE_WEIGHTS_FILE} ${INSTALL_DIR}

exit 0
