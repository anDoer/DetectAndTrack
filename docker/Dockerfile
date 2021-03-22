# Use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

RUN mv /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}
ENV CMAKE_INSTALL_PREFIX /usr/local:${CMAKE_INSTALL_PREFIX}
ENV CMAKE_MODULE_PATH /usr/local/cmake:${CMAKE_MODULE_PATH}
RUN apt-get -y update
RUN echo ASDasd
# Clone the Detectron repository
RUN git clone https://github.com/anDoer/DetectAndTrack.git /detectandtrack
RUN git clone https://github.com/gflags/gflags /opt/gflags
WORKDIR /opt/gflags
RUN mkdir build
WORKDIR /opt/gflags/build
RUN cmake ..
RUN make
RUN make install

RUN apt-get install -y libeigen3-dev python-tk wget

WORKDIR /opt
RUN wget https://github.com/google/glog/archive/v0.3.5.tar.gz
RUN tar xfz v0.3.5.tar.gz
WORKDIR /opt/glog-0.3.5/build
RUN cmake ..
RUN make && make install

# Install Python dependencies
#RUN pip install --upgrade pip
RUN cd /detectandtrack \
        && pip install -r requirements.txt

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Go to Densepose root
WORKDIR /detectandtrack/lib

# Set up Python modules
RUN make

# [Optional] Build custom ops
#RUN make ops

# Go to Densepose root
WORKDIR /detectandtrack

RUN apt-get install -y vim
#RUN apt-get clean

