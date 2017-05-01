import Fake_FacialDetection
import Predict
import sys

imagePath = sys.argv[1]

num = Fake_FacialDetection.face_dect_save(imagePath)
print (num)
Predict.make_prediction(num)
