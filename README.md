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

## Download the TensorFlow Lite Pre-trained model

```bash
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```

After downloading the model, you need to unzip it, use the below command for that to unzip and store it in a directory called model

```bash
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d model
```

## TODO List

- [x] Object Detection on Webcam
- [x] Object Detection on Video
- [ ] Object Detection on Image
- [ ] Object Detection using Custom models
