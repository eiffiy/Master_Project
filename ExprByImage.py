import FacialDetectionAndCrop
from Predict import PredictClass
import sys

imagePath = sys.argv[1]

num, folderName = FacialDetectionAndCrop.FaceCrop(imagePath)
print (num, folderName)
Pred = PredictClass()
Pred.makePredictionFromFolder(num, folderName)
