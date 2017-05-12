import cv2
import dlib
import numpy
import sys

PREDICTOR_PATH = "/Users/eiffiy/Master_Project/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class NoFaces(Exception):
    pass

im = cv2.imread("/Users/eiffiy/Master_Project/facial-expression-analyzed.jpg")

rects = detector(im, 1)

if len(rects) >= 1:
    print("{} faces detected".format(len(rects)))

if len(rects) == 0:
    raise NoFaces

for i in range(len(rects)):

    landmarks = numpy.matrix([[p.x, p.y]
                              for p in predictor(im, rects[i]).parts()])
    im = im.copy()

    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])

        cv2.circle(im, pos, 3, color=(0, 255, 0))

cv2.namedWindow("im", 2)
cv2.imshow("im", im)
cv2.waitKey(0)
