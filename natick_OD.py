"""
NATICK_OD.PY - Script for real-time object detection for Project Natick livestream
Code adapted from https://github.com/tensorflow/models/tree/master/research/object_detection
and from https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/
2018.08.15 Nile Wilson
"""
# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import tkinter as tk
import requests
import threading
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from imutils.video import FPS
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
from tkinter import *
from tkinter.filedialog import askopenfilename
from datetime import datetime

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# Model to download and/or import
MODEL_NAME = 'ssd_mobilenet_v2_fish'

# Path to frozen detection graph. This is the actual model that is used for the object detection
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box
PATH_TO_LABELS = os.path.join('data', 'pascal_label_map.pbtxt')
NUM_CLASSES = 2

# Load a (frozen) Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Threading
push_count = 0
class myThread(threading.Thread):
    def __init__(self, image_np, boxes, classes, scores, min_threshold, fps):
        self.image_np = image_np
        self.boxes = boxes
        self.classes = classes
        self.scores = scores
        self.min_threshold = min_threshold
        self.fps = fps  
    def push_data(self):
        # Prepare data to Power BI dashboard
        show_boxes = self.scores > self.min_threshold
        count = len(show_boxes[show_boxes])
        #print('Number of fish: ' + str(count))
        
        # Format data to send as JSON
        now = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%Z")
        if count == 0:
            score_avg = 0
            score_std = 0
            score_median = 0
            score_min = 0
            score_max = 0
        else:
            score_avg = np.mean(self.scores[self.scores > self.min_threshold])
            score_std = np.std(self.scores[self.scores > self.min_threshold])
            score_median = np.median(self.scores[self.scores > self.min_threshold])
            score_min = np.min(self.scores[self.scores > self.min_threshold])
            score_max = np.max(self.scores[self.scores > self.min_threshold])
            
        data = '[{{ "timestamp": "{0}", "count": "{1:d}", "score_avg": "{2:0.5f}", "score_std": "{3:0.5f}", "score_median": "{4:0.5f}", "score_min": "{5:0.5f}", "score_max": "{6:0.5f}", "min_threshold": "{7:0.5f}" }}]'.format(now, count, score_avg, score_std, score_median, score_min, score_max, self.min_threshold)

        # Send the data to the Power BI dashboard
        binary_data = data.encode('utf8')
        
        # Limit the amount of times data can be pushed per second
        max_pushes_per_second = 4
        time_seconds = time.time() % 60
        remainder = (time_seconds - np.floor(time_seconds)) % (1/max_pushes_per_second)

        # Reset count limit every second (leave some jitter room)
        jitter = 0.1
        global push_count
        if (time_seconds - np.floor(time_seconds)) < jitter*2:
          push_count = 0
        
        # If within jitter (0.1 seconds) of the timepoints where we can send data (if max_pushes_per_second = 4, then we can push at 0.25, 0.5, 0.75, and at 0)
        if (remainder <= jitter) or (abs(remainder - (1/max_pushes_per_second)) <= jitter):
          push_count += 1
          if push_count <= max_pushes_per_second:
            try:
              response = requests.post(REST_API_URL, data=binary_data)
              #print(data)
              #print(push_count)
            except requests.ConnectionError as e:
              print('[ERROR] Connection Error')
              print(str(e))
        

# REST API endpoint, given to you when you create an API streaming dataset
# Follow the tutorial here https://docs.microsoft.com/en-us/power-bi/service-real-time-streaming
# Will be of the format: https://api.powerbi.com/beta/<tenant id>/datasets/< dataset id>/rows?key=<key id>
REST_API_URL = "YOUR POWER BI URL HERE"

# Simple window to display if visualization is turned off (don't re-define in loop each frame)
img_blank = np.zeros((512,512,3), np.uint8)
cv2.putText(img_blank, 'Please check the Power BI dashboard', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255),1)

# Object Detection on video frames
class videoStreamer:
  def __init__(self):
    self.videoFile = 0
    self.visualization = True
  
  def useVideoFile(self):
    self.videoFile = askopenfilename()
    self.streamVideo()
  
  def useWebcam(self):
    self.videoFile = 0
    self.streamVideo()

  def useWebStream(self):
    #self.videoFile = 'http://natickmediaservices.streaming.mediaservices.windows.net/436cbbc1-6c6f-40e1-a3b3-f65baa4ecdc9/41f19f22-1154-4661-8d54-5adaf375d43a.ism/manifest(format=m3u8-aapl-v3)'
    self.videoFile = 'http://natickmediaservices.streaming.mediaservices.windows.net/57320b1f-7365-436c-8e4e-9bad1345e849/a5f06021-0b1e-4af9-8071-89727c774501.ism/manifest(format=m3u8-aapl-v3)'
    self.streamVideo()

  def select_visualization(self):
    if var2.get() == 1:
      selection = 'Video with object detection will be displayed'
      self.visualization = True
    elif var2.get() == 2:
      selection = 'No video will be displayed'
      self.visualization = False

    label2.config(text = selection)
  
  def streamVideo(self):
    """
    Note: The first frame read in through OpenCV is approximately 30 seconds behind the live video stream
    """
    # Access webcam or file
    cap = cv2.VideoCapture(self.videoFile)
    fps = FPS().start()

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        while True:
          ret, image_np = cap.read()

          if image_np is not None:
            # Resize captured frame if smallest dimension (excluding color) is greater than 300 px
            if min(image_np.shape[0:1]) > 300:
                reduceBy = max(image_np.shape[0:1])/300
                w_new = int(image_np.shape[0]/reduceBy)
                h_new = int(image_np.shape[1]/reduceBy)
                image_np = cv2.resize(image_np, dsize=(w_new,h_new), interpolation=cv2.INTER_CUBIC)
            
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            min_threshold = 0.5 # is 0.5 by default in visualize_boxes_and_labels_on_image_array
            
            if self.visualization == True:
                myThread0 = myThread(image_np, boxes, classes, scores, min_threshold, fps)

                # Push data to Power BI dashboard
                t2 = threading.Thread(target = myThread0.push_data)
                t2.daemon = True
                t2.start()

                # Non-threaded visualization
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=999,
                    min_score_thresh=min_threshold,
                    line_thickness=2)
                cv2.imshow(' Project Natick Environmental Sustainability Console - Realtime', cv2.resize(image_np, (800,600)))
                fps.update()

            elif self.visualization == False:
                myThread0 = myThread(img_blank, boxes, classes, scores, min_threshold, fps)
                t2 = threading.Thread(target = myThread0.push_data)
                t2.daemon = True
                t2.start()
                cv2.imshow(' Project Natick Environmental Sustainability Console - Realtime', cv2.resize(img_blank, (800, 600)))
                fps.update()

            # Exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                fps.stop()
                print('[INFO] elapsed time: {:.2f}'.format(fps.elapsed()))
                print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
                cap.release()
                cv2.destroyAllWindows()
                break


# Select whether to use webcam or file to run object detection on
streamer = videoStreamer()

# Create the GUI
# Create the window
root = tk.Tk(className=" Project Natick Environmental Sustainability Console")
root.geometry('900x500')
tmp_win = tk.Frame(root)
tmp_win.pack()

# Display how to quit
var = StringVar()
label = Message(tmp_win, textvariable=var)
label.config(font=('Courier', 24), justify=CENTER)
var.set('To quit Natick Console, press "q"')
label.pack(expand=YES, fill=BOTH)

# Create webcam button
button1 = tk.Button(tmp_win, text='Use Local Webcam', command=streamer.useWebcam)
button1.config(font=('Courier', 16), justify=CENTER, height=7, width=20, padx=10, bg='gray', fg='white')
button1.pack(side=LEFT)

# Create web stream button
button2 = tk.Button(tmp_win, text='Stream from Remote Natick Camera', command=streamer.useWebStream)
button2.config(font=('Courier', 16), justify=CENTER, height=7, width=20, padx=10, bg='gray', fg='white',wraplength=200)
button2.pack(side=LEFT)

# Create video file button
button3 = tk.Button(tmp_win, text='Use Local Video File', command=streamer.useVideoFile)
button3.config(font=('Courier', 16), justify=CENTER, height=7, width=20, padx=10, bg='gray', fg='white',wraplength=200)
button3.pack(side=LEFT)

# Create radio button to select visualization on/off
var2 = IntVar()
R1 = tk.Radiobutton(root, text='Visualization of Object Detection On', variable=var2, value=1, command=streamer.select_visualization)
R1.pack(anchor=W)

R2 = tk.Radiobutton(root, text='Visualization of Object Detection Off', variable=var2, value=2, command=streamer.select_visualization)
R2.pack(anchor=W)

label2 = Label(root)
label2.config(font=(cv2.FONT_HERSHEY_SIMPLEX, 14), justify=CENTER)
label2.pack(expand=YES, fill=BOTH)

# Keep the window open until user closes it
root.mainloop()