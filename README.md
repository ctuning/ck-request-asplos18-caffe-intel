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

### Install Caffe models

#### ResNet50

```
$ ck install package:caffemodel-resnet50
``` 

#### Inception v3

**TODO**

#### SSD

```
$ ck install package:caffemodel-ssd-voc-300 
```

### Install ImageNet validation datasets

#### Reduced (500 images)
```
$ ck install package:imagenet-2012-val-min
```

#### Full (50,000 images)
```
$ ck install package:imagenet-2012-val
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
