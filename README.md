# Collective Knowledge workflow for the Intel Caffe submission to [ReQuEST @ ASPLOS'18](http://cknowledge.org/request-cfp-asplos2018.html)

- [Authors' instructions](https://github.com/intel/caffe/wiki/ReQuEST-Artifact-Installation-Guide)

## Installation instructions

### Install Collective Knowledge

```
$ sudo pip install ck
```

### Install Intel Caffe from the ReQuEST artifact branch

```
$ ck pull repo:ck-request-asplos18-caffe-intel
$ ck install package:lib-caffe-intel-request-cpu
```

### Install ImageNet validation datasets

**NB:** If you already have the ImageNet validation dataset downloaded, e.g. in
`/datasets/ilsvrc2012_val/`, you can simply register it with CK as follows:

```
$ ck detect soft:dataset.imagenet.val \
--full_path=/datasets/ilsvrc2012_val/ILSVRC2012_val_00000001.JPEG
```

#### Reduced (500 images)
```
$ ck install package:imagenet-2012-val-min
```

#### Full (50,000 images)
```
$ ck install package:imagenet-2012-val
```

### Install Caffe models and resize ImageNet dataset

**NB:** If you already have the ImageNet validation dataset resized as an LMDB file, e.g. in `/datasets/dataset-imagenet-ilsvrc2012-val-lmdb-dataset.imagenet.val-ilsvrc2012_val_full-resize-320/data/data.mdb`, you can register it with CK as follows:

````
$ ck detect soft:dataset.imagenet.val.lmdb \
--full_path=/datasets/dataset-imagenet-ilsvrc2012-val-lmdb-dataset.imagenet.val-ilsvrc2012_val_full-resize-320/data/data.mdb
```

#### ResNet50

**NB:** ResNet50 uses the standard ImageNet mean file of resolution `256x256`, so the inputs must match that.

```
$ ck install ck-caffe:package:imagenet-2012-val-lmdb-256
$ ck install ck-caffe:package:caffemodel-resnet50
$ ck install ck-request-asplos18-caffe-intel:package:caffemodel-resnet50-intel-i8
```

#### Inception-v3

**NB:** Inception-v3 uses an ImageNet mean file of resolution `320x320`, so the inputs must match that.

```
$ ck install ck-caffe:package:imagenet-2012-val-lmdb-320
$ ck install ck-caffe:package:caffemodel-inception-v3
$ ck install ck-request-asplos18-caffe-intel:package:caffemodel-inception-v3-intel-i8
```

#### SSD

```
$ ck install package:caffemodel-ssd-voc-300
```


## Usage instructions

### Measure latency
```
$ ck run program:caffe --cmd_key=time_cpu --env.CK_CAFFE_BATCH_SIZE=1
```

### Measure throughput
```
$ ck run program:caffe --cmd_key=time_cpu --env.CK_CAFFE_BATCH_SIZE=64
```

### Measure accuracy
```
$ ck run program:caffe --cmd_key=test_cpu
```
