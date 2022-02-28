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
    # print(result.detections)
    if result.detections:
        for id, detect in enumerate(result.detections):
            # mp_draw.draw_detection(img, detect)
            # print(id)
            # print(detect.score)
            # print(id, detect)
            # print(detect.location_data.relative_bounding_box)
            bound_box_class = detect.location_data.relative_bounding_box
            img_height, img_width, img_channel = img.shape
            bound_box = int(bound_box_class.xmin * img_width), \
                        int(bound_box_class.ymin * img_height), \
                        int(bound_box_class.width * img_width), \
                        int(bound_box_class.height * img_height)
            cv2.rectangle(
                img, bound_box,
                (255, 0, 255), 2
            )

    cv2.imshow("Image", img)
    cv2.waitKey(1)
