#!/bin/bash

# Get packages required for OpenCV
sudo apt install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev -y
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo apt install libxvidcore-dev libx264-dev -y
sudo apt install qt4-dev-tools libatlas-base-dev -y
sudo apt install libilmbase-dev libopenexr-dev libgstreamer1.0-dev -y

# Need to get an older version of OpenCV because version 4 has errors
pip3 install opencv-python

# Get packages required for TensorFlow
# pip3 install tensorflow (uncomment to install tensorflow)

# To install tflite runtime
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime