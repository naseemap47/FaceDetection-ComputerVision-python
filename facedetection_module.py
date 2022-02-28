import cv2
import mediapipe as mp
import time

def detectfaces(
    display_fps=True, p_time=0,
    min_detection_confidence=0.5,
    default_draw=False
):
    cap = cv2.VideoCapture(0)
    mp_face_detect = mp.solutions.face_detection
    face_detect = mp_face_detect.FaceDetection(min_detection_confidence=min_detection_confidence)
    
    mp_draw = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_detect.process(img_rgb)
        # print(result.detections)
        if result.detections:
            for id, detect in enumerate(result.detections):
                if default_draw:
                    mp_draw.draw_detection(img, detect)
                # print(id)
                # print(detect.score)
                # print(id, detect)
                # print(detect.location_data.relative_bounding_box)
                if default_draw is False:
                    bound_box_class = detect.location_data.relative_bounding_box
                    img_height, img_width, img_channel = img.shape
                    bound_box = int(bound_box_class.xmin * img_width), \
                                int(bound_box_class.ymin * img_height), \
                                int(bound_box_class.width * img_width), \
                                int(bound_box_class.height * img_height)
                    cv2.rectangle(
                        img, bound_box,
                        (0, 255, 0), 1
                    )
                    cv2.putText(
                        img, f'{int(detect.score[0] * 100)}%',
                        (bound_box[0], bound_box[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        (0,255,0), 1
                    )
        if display_fps:
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time

            cv2.putText(
                img, f'FPS: {int(fps)}', (10,60),
                cv2.FONT_HERSHEY_PLAIN, 2,
                (255,0,0), 2
            )
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)