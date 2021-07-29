#!/usr/bin/python3

import os
import cv2
import argparse
import importlib.util

import numpy as np
from threading import Thread

# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream():
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self, resolution=(640, 480)):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        _ = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        _ = self.stream.set(3,resolution[0])
        _ = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        self.grabbed, self.frame = self.stream.read()

	    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            self.grabbed, self.frame = self.stream.read()

    def read(self):
	    # Return the most recent frame
        return self.frame

    def stop(self):
	    # Indicate that the camera and thread should be stopped
        self.stopped = True

def detect(videostream, frame_rate_calc):
    while True:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        frame1 = videostream.read()

        # Acquire frame and resize
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Passing Input Image to the pre-trained model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Getting the output info
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        # Draw bounding boxes for every prediction
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if score > min_conf_threshold and score <= 1.0:
                # Getting bounding box dimensions
                ymin = int(max(1, (box[0] * imH)))
                xmin = int(max(1, (box[1] * imW)))
                ymax = int(min(imH, (box[2] * imH)))
                xmax = int(min(imW, (box[3] * imW)))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw Label
                object_name = labels[int(cls)]
                label = f"{object_name} {round(score*100, 2)}%"
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        cv2.putText(frame, f"FPS: {round(frame_rate_calc, 2)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Object Detection (q to close)', frame)

        t2 = cv2.getTickCount()
        time1 = (t2 - t1)/freq
        frame_rate_calc = 1/time1

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    videostream.stop()

# Define and parse input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', help='Path of directory which contains .tflite file', required=True)
parser.add_argument('--tflite_file', help='Name of .tflite if different from default', default='detect.tflite')
parser.add_argument('--labels', help='Name of labels file if different from default', default='labelmap.txt')

args = parser.parse_args()

MODEL_NAME = args.model_dir
GRAPH_NAME = args.tflite_file
LABELMAP_NAME = args.labels

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS) as fh:
    labels = [line.strip() for line in fh.readlines()]

if labels[0] == '???':
    del(labels[0])

interpreter = Interpreter(model_path=PATH_TO_CKPT)

min_conf_threshold = 0.5
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

# Image Preprocessing dimensions
input_mean = 127.5
input_std = 127.5

# Initialize video stream
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Window resolution
imW = 1280
imH = 720

if __name__ == "__main__":
    # Intialize video stream
    videostream = VideoStream(resolution=(imW, imH)).start()
    detect(videostream, frame_rate_calc)