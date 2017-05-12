import cv2
import sys
import dlib
import numpy
import logging as log
import datetime as dt
from time import sleep

video_capture = cv2.VideoCapture(0)
anterior = 0
time_mark = 0

PREDICTOR_PATH = "/Users/eiffiy/Master_Project/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(PREDICTOR_PATH)

while True:
    print(time_mark)
    time_mark = time_mark + 1
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, im = video_capture.read()

    rects = detector(im, 1)

    for i in range(len(rects)):
        landmarks = numpy.matrix([[p.x, p.y]
                                  for p in predictor(im, rects[i]).parts()])
        im = im.copy()

        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(im, pos, 3, color=(0, 255, 0))

    # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', im)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
