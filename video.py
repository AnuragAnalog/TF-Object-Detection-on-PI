#!/usr/bin/python3

import os
import cv2
import argparse
import importlib.util

import numpy as np

def detect(video, frame_rate_calc):
    while video.isOpened():
        ret, frame1 = video.read()
        if not ret:
          print('Reached the end of the video!')
          break

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

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
    video.release()

# Define and parse input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', help='Path of directory which contains .tflite file', required=True)
parser.add_argument('--tflite_file', help='Name of .tflite if different from default', default='detect.tflite')
parser.add_argument('--labels', help='Name of labels file if different from default', default='labelmap.txt')
parser.add_argument('--video', help='Name of the video path', default='video.mp4')

args = parser.parse_args()

MODEL_NAME = args.model_dir
GRAPH_NAME = args.tflite_file
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

CWD_PATH = os.getcwd()

VIDEO_PATH = os.path.join(CWD_PATH, VIDEO_NAME)
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

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

if __name__ == "__main__":
    # Intialize video stream
    detect(video, frame_rate_calc)