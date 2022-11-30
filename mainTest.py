import cv2 as cv
from keras.models import load_model
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model = load_model("dianseAIepochs10.h5")
# img = cv.imread("dataset\no\no0.jpg")
print(type(model))

img = cv.imread("C:\\Users\\avni\\Desktop\\projects\ML projects\\diagnose-ai\\dataset\\pred\\pred5.jpg")
# cv.imshow("img", img2)
img_arr = Image.fromarray(img)
print(type(img_arr))

resized_img = img_arr.resize((64, 64))
final_img = np.array(resized_img)

# print(final_img)

input_img = np.expand_dims(final_img, axis=0)
print(type(input_img))

result = model.predict(input_img)

print(result)


