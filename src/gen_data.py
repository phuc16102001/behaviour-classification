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
import pandas as pd
import time

from scipy.misc import face

# Init variables
cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()
ls_landmark = []

label = input("Class name:")
nFrame = 600
countDown = 10

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
def draw_landmark(frame, mpDraw, pose_landmarks=None, face_landmarks = None):
    if (pose_landmarks is not None):
        mpDraw.draw_landmarks(frame, pose_landmarks, mpPose.POSE_CONNECTIONS)
    if (face_landmarks is not None):
        mpDraw.draw_landmarks(frame, face_landmarks, mpPose.FACEMESH_CONTOURS)
    return frame

def draw_count_frame(cnt, total, frame):
    text = "Frame: {}/{}".format(cnt, total)
    pos = (10,30)
    scale = 1
    thickness = 2
    lineType = 2
    fontColor = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        text,
        pos,
        font,
        scale,
        fontColor,
        thickness,
        lineType
    )
    return frame

for i in range(countDown):
    print(i)
    time.sleep(1)

while len(ls_landmark)<nFrame:
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
        img = frame.copy()
        if (poseRet.pose_landmarks):
            landmark = make_landmark_timestamp(poseRet)
            ls_landmark.append(landmark)
            img = draw_landmark(frame, mpDraw, pose_landmarks=poseRet.pose_landmarks, face_landmarks = None)

        # Draw frame count
        img = draw_count_frame(len(ls_landmark), nFrame,img)

        # Show pose
        cv2.imshow('pose', img)
        
df = pd.DataFrame(ls_landmark)
df.to_csv("data/{}.csv".format(label),index=False)

cap.release()
cv2.destroyAllWindows()