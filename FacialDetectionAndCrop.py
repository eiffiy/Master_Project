import cv2
import sys
import os

import PIL
from PIL import Image


def newFolder(name):
    if not os.path.isdir(name):
        try:
            os.makedirs(name)
        except OSError:
            pass
        # let exception propagate if we just can't
        # cd into the specified directory
        # os.chdir(path)

# input the img file dir


def FaceCrop(path):

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
    )

    print("Found {0} faces!".format(len(faces)))

    # create Image object
    img = Image.open(imagePath)

    # get the name of the input img
    name = path.split('/')[(len(path.split('/')) - 1)].split('.')
    # create a new folder that called img's name for cropped imgs
    newFolder('./CroppedImgs/' + name[0])

    # account the number of faces
    i = 0

    # Draw a rectangle around the faces and crop them
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_crop = img.crop((x, y, x + w, y + h))
        img_crop.thumbnail((48, 48))
        # save the cropped img into CroppedImgs/ImgName folder
        img_crop.save('./CroppedImgs/' +
                      name[0] + '/' + str(i) + '.jpg', "JPEG")
        i = i + 1

    cv2.imshow("Faces found", image)
    # print ("##################")
    cv2.waitKey(0)
    return [i, name[0]]
