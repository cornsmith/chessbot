FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
	build-essential \
	cmake \
	git \
	libgtk2.0-dev \
	libjasper-dev \
	libjpeg-dev \
	libpng-dev \
	libswscale-dev \
	libtbb-dev \
	libtbb2 \
	libtiff-dev \
	pkg-config \
	python3 \
	python3-dev \
	python3-numpy \
	python3-pip \
	unzip \
	wget

RUN pip3 install --upgrade pip && pip3 install scipy scikit-learn python-chess

RUN cd && \
	wget https://github.com/opencv/opencv/archive/3.1.0.zip && \
	unzip 3.1.0.zip && \
	cd opencv-3.1.0 && \
	mkdir build && \
	cd ~/opencv-3.1.0/build && \
	cmake .. && \
	make -j4 && \
	make install && \
	cd && \
	rm 3.1.0.zip
