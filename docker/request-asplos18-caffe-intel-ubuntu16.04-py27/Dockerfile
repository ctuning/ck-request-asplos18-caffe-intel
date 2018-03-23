FROM ubuntu:16.04
MAINTAINER Grigori Fursin <Grigori.Fursin@cTuning.org>

# Install standard packages.
RUN apt-get update && apt-get install -y \
    python-all \
    python-pip \
    git zip bzip2 sudo wget \
    libglib2.0-0 libsm6

RUN pip install ck
RUN ck  version

# Install CK RPi repo
RUN ck pull repo:ck-request-asplos18-caffe-intel
RUN ck pull repo:ck-request-asplos18-results-caffe-intel

#
CMD bash
