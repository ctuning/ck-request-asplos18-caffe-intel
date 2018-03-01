# convert-ssd-to-i8

Program for conversion of SSD models into Intel Caffe quantized format.

## Requirements

Intel Caffe library:
```
ck install package --tags=lib,caffe,intel
```

Caffe model to be converted into quantized format:
```
ck install package --tags=caffemodel,ssd
```

One of datasets COCO-2014 or VOC-2007 depending on which model you wish to convert:
```
ck install package:dataset-coco-2014
ck install package:dataset-voc-2007
```

## Run

```
ck run program:convert-ssd-to-i8 --cmd_key=convert-coco
```
or
```
ck run program:convert-ssd-to-i8 --cmd_key=convert-voc
```

Program takes dataset and prepares LMDB database. Number of images to be included into database is governed by variable `CK_IMAGES_PERCENT`.

Resulting models are stored into `tmp` directory as `test_quantized.prototxt` and `deploy_quantized.prototxt`.

## Test

Program also provide two testing commands:
```
ck run program:convert-ssd-to-i8 --cmd_key=test-f32
ck run program:convert-ssd-to-i8 --cmd_key=test-i8
```
Commands take LMDB database that was prepared at previous `convert` stage and run `caffe test` operaion using original or converted models.

**TODO**
We should somehow interpret command output. It produce stdout like
```
I0228 11:02:49.898972 63713 caffe.cpp:472] Running for 171 iterations.
I0228 11:03:08.554591 63713 caffe.cpp:439]     Test net output #0: detection_eval = 0
```
and original Intel's `benchmark.py` script treats `detection_eval` value as mAP.