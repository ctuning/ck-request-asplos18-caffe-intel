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
  echo "Converting ${INPUT_PROTOTXT} ..."
  echo

  if [ -f ${INSTALL_DIR}/${INPUT_PROTOTXT} ]; then
    rm /tmp/${INPUT_PROTOTXT}
  fi
  if [ -f ${INSTALL_DIR}/${OUTPUT_PROTOTXT} ]; then 
    rm ${INSTALL_DIR}/${OUTPUT_PROTOTXT}
  fi

  python ${THIS_SCRIPT_DIR}/prepare_prototxt.py \
    --model=${ORIGINAL_PACKAGE_DIR}/${INPUT_PROTOTXT} \
    --target=/tmp/${INPUT_PROTOTXT}

  python ${CK_ENV_LIB_CAFFE}/../src/scripts/calibrator.py \
    --root=${CK_ENV_LIB_CAFFE} \
    --weights=${CK_ENV_MODEL_CAFFE_WEIGHTS} \
    --model=/tmp/${INPUT_PROTOTXT} \
    --iterations=${ITERATIONS} \
    --blob_name=${BLOB_NAME}

  if [ ${PROTOTXT} == "deploy" ]; then
    FINALIZE_SCRIPT=finalize_deploy_prototxt.py
  else
    FINALIZE_SCRIPT=finalize_train_val_prototxt.py
  fi
  python ${THIS_SCRIPT_DIR}/${FINALIZE_SCRIPT} \
    --model=/tmp/${INPUT_PROTOTXT} \
    --target=${INSTALL_DIR}/${OUTPUT_PROTOTXT}

  rm /tmp/${INPUT_PROTOTXT}
  rm /tmp/${OUTPUT_PROTOTXT}
}

# At this path, there should be a module that can be imported.
#   from caffe.proto import caffe_pb2
# This is required in the calibration.py script.
PYTHON_CAFFE_INIT_PY=${CK_ENV_LIB_CAFFE}/python/caffe/__init__.py
if [ ! -f ${PYTHON_CAFFE_INIT_PY} ]; then
  PYTHON_CAFFE_INIT_PY_CREATED="YES"
  echo '' > ${PYTHON_CAFFE_INIT_PY}
else
  PYTHON_CAFFE_INIT_PY_CREATED="NO"
fi

ConvertPrototxt 'train_val' ${TRAIN_VAL_BLOB_NAME} ${TRAIN_VAL_ITERATIONS}
ConvertPrototxt 'deploy' ${DEPLOY_BLOB_NAME} ${DEPLOY_ITERATIONS}

cp ${CK_ENV_MODEL_CAFFE}/${CK_ENV_MODEL_CAFFE_MEAN_BIN_FILE} ${INSTALL_DIR}
cp ${CK_ENV_MODEL_CAFFE}/${CK_ENV_MODEL_CAFFE_WEIGHTS_FILE} ${INSTALL_DIR}

if [ ${PYTHON_CAFFE_INIT_PY_CREATED} == "YES" ]; then
  rm ${PYTHON_CAFFE_INIT_PY}
  rm ${PYTHON_CAFFE_INIT_PY}c
fi

exit 0
