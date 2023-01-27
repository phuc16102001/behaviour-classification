'''
This is a project for behaviour classification
The project is built to demonstrate for the research methodology course
Owner: Do Vuong Phuc
Reference: Mi AI - Nhan dien hanh vi con nguoi

PLEASE DO NOT COPY WITHOUT PERMISSION!
'''

# Libraries
import cv2
import mediapipe as mp
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import threading
from keras.models import load_model
from config import *

# Init variables
cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

model = load_model('../models/best.h5')
ls_landmark = []
label = "None"

# Create dataset of landmarks and timestamp
def make_landmark_timestamp(poseRet):
    ret = []
    for idx, lm in enumerate(poseRet.pose_landmarks.landmark):
        ret.append(lm.x)
        ret.append(lm.y)
        ret.append(lm.z)
        ret.append(lm.visibility)
    return ret

# Draw landmarks on image
def draw_landmark(mpDraw, poseRet, frame):
    mpDraw.draw_landmarks(frame, poseRet.pose_landmarks, mpPose.POSE_CONNECTIONS)
    return frame

def draw_label(label, frame):
    text = "Class: {}".format(label)
    pos = (10,30)
    scale = 1
    thickness = 2
    lineType = 2
    fontColor = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
                text,
                pos,
                font,
                scale,
                fontColor,
                thickness,
                lineType)
    return frame

def detect(model, ls_landmark):
    global label
    tensor = np.expand_dims(ls_landmark,axis=0)
    result = model.predict(tensor)
    label = classes[np.argmax(result[0])]
    print(np.round(np.array(result[0]),2))

# Extract classes
files = os.listdir('../data')
classes = []
for path in files:
    classes.append(path.split('.')[0])
list.sort(classes)

while True:
    ret, frame = cap.read()
    if (ret):
        # Show input
        cv2.imshow('camera', frame)
        if cv2.waitKey(1)==ord('q'):
            break
        
        # Convert to RGB and create pose estimation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        poseRet = pose.process(rgb)

        # Draw and create data
        if (poseRet.pose_landmarks):
            landmark = make_landmark_timestamp(poseRet)
            ls_landmark.append(landmark)
            frame = draw_landmark(mpDraw, poseRet, frame)

        # Inference
        if (len(ls_landmark)==N_TIME):
            t = threading.Thread(
                target = detect,
                args = (model, ls_landmark)
            )
            t.start()
            ls_landmark = []

        # Draw frame count
        frame = draw_label(label, frame)

        # Show pose
        cv2.imshow('pose', frame)

cap.release()
cv2.destroyAllWindows()