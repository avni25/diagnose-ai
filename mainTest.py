import cv2 as cv
from keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("dianseAIepochs10.h5")
img = cv.imread("\dataset\no\no0.jpg")
cv.imshow(img)
img_arr = Image.fromarray(img, "RGB")

resized_img = img_arr.resize((64, 64))
final_img = np.array(resized_img)









cv.waitKey(0)