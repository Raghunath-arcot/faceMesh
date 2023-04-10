import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

facemesh = mp.solutions.face_mesh
face = facemesh.FaceMesh(static_image_mode=True, min_tracking_confidence=0.5, min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils

while True:

    _, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    op = face.process(rgb)
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
            print(i.landmark[0].y*480)
            draw.draw_landmarks(frame, i, facemesh.FACEMESH_CONTOURS, landmark_drawing_spec=draw.DrawingSpec(color= (50, 0, 0), circle_radius= 0))

    cv2.imshow("window", frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
