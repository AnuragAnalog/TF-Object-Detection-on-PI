# Tensorflow Lite Object Detetion on Raspberry PI

To run the tensorflow lite model on your raspberry pi, do follow the below steps:

## Hardware Requirements

* Raspberry Pi
* PI Camera Module

![PI with camera](./pi-with-camera.jpg)

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

## Code

I have written two varaints of the program, one which takes input from your wecam feed and the other which takes input from a video file.

> For video.py I have added a sample [video file](https://www.indiavideo.org/)

## Run the program

To run the webcam version

```bash
./webcam.py --model_dir=model
```

To run the video version

```bash
./video.py --model_dir=model
```

## Demo

![This is how I look :P](./preview.png)

## TODO List

- [x] Object Detection on Webcam
- [x] Object Detection on Video
- [ ] Object Detection on Image
- [ ] Add more links to pre-trained models
- [ ] Object Detection using Custom models
