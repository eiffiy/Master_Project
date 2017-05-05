import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import PIL
from PIL import Image
import Predict

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
timer = 0

while True:

    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    timer = timer + 1

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if timer % 5 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            img = Image.fromarray(frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img = img.crop((x, y, x + w, y + h))
            img = img.convert('L')
            img.thumbnail((48, 48))
            end_label = Predict.make_prediction_BycroppedImg(img)

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
