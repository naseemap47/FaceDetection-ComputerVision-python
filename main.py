from unittest import result
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_face_detect = mp.solutions.face_detection
face_detect = mp_face_detect.FaceDetection()

mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_detect.process(img_rgb)
    print(result.detections)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
