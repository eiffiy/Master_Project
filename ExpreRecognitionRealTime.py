import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import PIL
from PIL import Image
from Predict import PredictClass

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
timer = 0

# build model and this can accelerate calculation in later
Pred = PredictClass()

while True:

    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    timer = timer + 1

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if timer % 3 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(128, 128)
        )

        # Draw a rectangle around the faces
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Problem: if in a frame there are many faces,
        # one print in stable position is not enough
        for (x, y, w, h) in faces:
            print(w, h)
            img = Image.fromarray(frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img = img.crop((x, y, x + w, y + h))
            img = img.convert('L')
            img.thumbnail((48, 48))
            # according to cropped imgs to predict, Input a Image object
            str_label, Pre_lebal = Pred.makePredictionFromCam(img)
            # print expression on the screen
            cv2.putText(frame,
                        str_label,
                        (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1)

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: " + str(len(faces)) +
                     " at " + str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
