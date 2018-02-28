#!/usr/bin/env python
#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import json
import imp
import os
import shutil
import subprocess

from google.protobuf import text_format

########################################################################

# Should be more robust criterion, may be lib should proivde some env var
def model_img_w(model_path): return 300 if '-300' in model_path else 512
def model_img_h(model_path): return 300 if '-300' in model_path else 512

########################################################################

def run_command(args_list):
  print(' '.join(args_list))
  process = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  output = process.communicate()[0]
  print(output)
  return output

########################################################################

# Despite of CK_ENV_LIB_CAFFE_PYTHON is in PYTHONPATH, we can't import caffe_pb2
# because of caffe.proto is not a package (at least in package:lib-caffe-ssd-cpu)
def import_caffe_pb2():
  caffe_python = os.getenv('CK_ENV_LIB_CAFFE_PYTHON')
  module_path = os.path.join(caffe_python, 'caffe', 'proto', 'caffe_pb2.py')
  return imp.load_source('caffe_pb2', module_path)

caffe_pb2 = import_caffe_pb2()

########################################################################

def read_json(file_name):
  with open(file_name, 'r') as f:
    return json.load(f)

def write_json(file_name, obj):
  with open(file_name, 'w') as f:
    json.dump(obj, f, indent=2, sort_keys=True)
    
########################################################################

def read_text(file_name):
  with open(file_name, 'r') as f:
    return f.read()

def write_text(file_name, txt):
  with open(file_name, 'w') as f:
    f.write(txt)

########################################################################

def read_prototxt(file_name):
  proto = caffe_pb2.NetParameter()
  txt = read_text(file_name)  
  text_format.Merge(txt, proto)
  return proto

def write_prototxt(file_name, proto):
  txt = text_format.MessageToString(proto)
  write_text(file_name, txt)

########################################################################

def rmdir(dir_name):
  if os.path.isdir(dir_name):
    shutil.rmtree(dir_name)

def mkdir(dir_name):
  if os.path.isdir(dir_name):
    shutil.rmtree(dir_name)
  os.mkdir(dir_name)

########################################################################

def prepare_test_prototxt(src_file, dst_file, lmdb_dir, batch_size,
                          label_map_file, name_size_file, image_count):
  '''
  Prepares test.prototxt file replacing ck-variables to their real values
  and substituting real paths to lmdb, label map, etc.
  '''
  net = caffe_pb2.NetParameter()
  txt = read_text(src_file)
  txt = txt.replace('$#val_batch_size#$', str(batch_size))
  txt = txt.replace('$#num_test_image#$', str(image_count))
  text_format.Merge(txt, net)
  for layer in net.layer:
    if layer.name == 'data':
      layer.data_param.source = lmdb_dir
      layer.data_param.batch_size = batch_size
      layer.annotated_data_param.label_map_file = label_map_file
    elif layer.name == 'detection_out':
      p = layer.detection_output_param.save_output_param
      p.label_map_file = label_map_file
      p.name_size_file = name_size_file
      p.num_test_image = image_count
    elif layer.name == 'detection_eval':
      layer.detection_evaluate_param.name_size_file = name_size_file
  write_prototxt(dst_file, net)
