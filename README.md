# Tensorflow Lite Object Detetion on Raspberry PI

To run the tensorflow lite model on your raspberry pi, do follow the below steps:

## Hardware Requirements

* Raspberry Pi
* Pi Camera Module

## Clone

Clone the repo onto your raspberry pi machine

```bash
git clone https://github.com/AnuragAnalog/TF-Object-Detection-on-PI
cd TF-Object-Detection-on-PI
```

## Download the required packages

Run the download.sh file to download all the required packages which are necessary to run opencv

```bash
./download.sh
```

> If you want to install tensorflow instead of tflite_runtime, you can comment the last four lines in the script.

## TODO List

- [x] Object Detection on Webcam
- [x] Object Detection on Video
- [ ] Object Detection on Image
- [ ] Object Detection using Custom models