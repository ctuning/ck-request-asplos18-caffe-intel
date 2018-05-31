[![logo](https://github.com/ctuning/ck-guide-images/blob/master/logo-powered-by-ck.png)](https://github.com/ctuning/ck)

This repository contains experimental workflow and all related artifacts 
as portable, customizable and reusable [Collective Knowledge components](https://github.com/ctuning/ck)
for image classification from the [1st ReQuEST tournament at ASPLOS'18](http://cknowledge.org/request-cfp-asplos2018.html) 
on reproducible SW/HW co-design of deep learning (speed, accuracy, energy, costs).

## References

* **Title:** Highly Efficient 8-bit Low Precision Inference of Convolutional Neural Networks with IntelCaffe
* **Authors:** Jiong Gong, Haihao Shen, Guoming Zhang, Xiaoli Liu, Shane Li, Ge Jin, Niharika Maheshwari

* [ACM paper](https://doi.org/10.1145/3229762.3229763)
* [ACM artifact](https://doi.org/10.1145/3229769)

* [arXiv ReQuEST goals](https://arxiv.org/abs/1801.06378)

## Artifact check-list (meta-information)

We use the standard [Artifact Description check-list](http://ctuning.org/ae/submission_extra.html) from systems conferences including CGO, PPoPP, PACT and SuperComputing.

* **Algorithm:** image classification
* **Program:** 
* **Compilation:** Intel C++ Compiler 17.0.5 20170817
* **Transformations:**
* **Binary:** will be compiled on a target platform
* **Data set:** ImageNet 2012 validation (50,000 images)
* **Run-time environment:** 
```
KMP HW SUBSET=1T
KMP AFFINITY=granularity=fine,compact
OMP NUM THREADS=18
```
* **Hardware:** single socket (18 cores) on c5.18xlarge
* **Run-time state:** 
* **Execution:**
* **Metrics:** 
```
Throughput: images per second.
Latency: milli-second.
Accuracy: % top-1/top-5/mAP.
```
* **Output:** classification result; execution time; accuracy
* **Experiments:** 

```
We use batch size 64, 64, and 32 to measure the
throughput for ResNet-50, Inception-V3, and SSD respectively.
We use batch size 1 to measure the latency.
```

* **How much disk space required (approximately)?** 
* **How much time is needed to prepare workflow (approximately)?** 
* **How much time is needed to complete experiments (approximately)?**
* **Collective Knowledge workflow framework used?** Yes
* **Publicly available?:** Yes
* **Experimental results:** https://github.com/ctuning/ck-request-asplos18-results-caffe-intel
* **Scoreboard:** http://cKnowledge.org/request-results

## Installation instructions

- [Authors' instructions](https://github.com/intel/caffe/wiki/ReQuEST-Artifact-Installation-Guide)

### Minimal CK installation

The minimal installation requires:

* Python 2.7 or 3.3+ (limitation is mainly due to unitests)
* Git command line client.

You can install CK in your local user space as follows:

```
$ git clone http://github.com/ctuning/ck
$ export PATH=$PWD/ck/bin:$PATH
$ export PYTHONPATH=$PWD/ck:$PYTHONPATH
```

You can also install CK via PIP with sudo to avoid setting up environment variables yourself:

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

```
$ ck detect soft:dataset.imagenet.val.lmdb \
--full_path=/datasets/dataset-imagenet-ilsvrc2012-val-lmdb-dataset.imagenet.val-ilsvrc2012_val_full-resize-320/data/data.mdb
```

#### ResNet50

**NB:** ResNet50 uses an ImageNet mean file of resolution `224x224`, so the inputs must match that.

```
$ ck install ck-caffe:package:imagenet-2012-val-lmdb-224
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

### Measure accuracy
```
$ ck run program:caffe --cmd_key=test_cpu
```

Results:
- https://github.com/ctuning/ck-request-asplos18-caffe-intel/issues/7#issuecomment-374265425
- https://github.com/ctuning/ck-request-asplos18-caffe-intel/issues/9#issuecomment-374268187


### Measure latency
```
$ ck run program:caffe --cmd_key=time_cpu --env.CK_CAFFE_BATCH_SIZE=1
```

### Measure throughput
```
$ ck run program:caffe --cmd_key=time_cpu --env.CK_CAFFE_BATCH_SIZE=64
```

### Explore performance

Explore how the execution time is affected by changing:
- [`nt`] the number of OpenMP threads (e.g. from 1 to 20 on a 10-core machine with hyperthreading);
- [`bs`] the batch size (e.g. from 1 to 64).

**NB:** You may want to change the `bs` and `nt` space exploration parameters, as well as
`platform_tags` in the `benchmarking.py` script before launching it as follows:

```
$ python `ck find script:explore-batch-size-openmp-threads`/benchmarking.py
```

## Unify output and add extra dimensions

Scripts to unify all experiments and add extra dimensions in ReQuEST format for further comparison and visualization are available in the following entry:
```
$ cd `ck find ck-request-asplos18-caffe-intel:script:explore-batch-size-openmp-threads`
```

- benchmark-merge-performance-with-accuracy.py - merges performance entries with accuracy
- benchmark-add-dimensions-*.py - adds extra dimensions for different platforms

CPU price is taken from [here](https://ark.intel.com/products/81705/Intel-Xeon-Processor-E5-2650-v3-25M-Cache-2_30-GHz).

All updated experimental results are then moved to [ck-request-asplos18-results-caffe-intel repository](https://github.com/ctuning/ck-request-asplos18-results-caffe-intel).
The best configurations are also moved to [ck-request-asplos18-results repo](https://github.com/ctuning/ck-request-asplos18-results).

## ReQuEST scoreboard

Best and reference results are now added to the [ReQuEST scoreboard](http://cKnowledge.org/request-results).

## Reviewers

This workflow was converted to CK and validated by the following reviewers:
* [Anton Lokhmotov](http://dividiti.com)
* [Flavio Vella](http://dividiti.com)
* [Grigori Fursin](http://fursin.net/research.html)
