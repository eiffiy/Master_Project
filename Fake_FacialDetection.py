import cv2
import sys

import PIL
from PIL import Image


def face_dect_save(path):
    # Get user supplied values
    imagePath = path
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    img = Image.open(imagePath)

    # Draw a rectangle around the faces
    i = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_crop = img.crop((x, y, x + w, y + h))
        img_crop.thumbnail((48, 48))
        img_crop.save(str(i) + '.jpg', "JPEG")
        i = i + 1

    cv2.imshow("Faces found", image)
    print ("##################")
    cv2.waitKey(0)
    return i
