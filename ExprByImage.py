import FacialDetectionAndCrop
import Predict
import sys

imagePath = sys.argv[1]

num = FacialDetectionAndCrop.FaceCrop(imagePath)
print (num)
Predict.make_prediction(num)
