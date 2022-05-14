from model import model
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import normalize

# 
X_test,y_test=[],[]

IMG_HEIGHT=256
IMG_WIDTH=256
IMG_CHANNELS=3
def get_model():
    return model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)



# #Predict on a few images
model = get_model()
model.load_weights('weights/weights.hdf5') 
# # 	# evaluate model
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

# # IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)
# #--------------------------------------------
test_img = cv2.imread('', 0)
test_mask = cv2.imread('', 0)

test_img_other = cv2.imread('', 0)
test_mask_other = cv2.imread('', 0)

test_img_norm = np.expand_dims(normalize(np.array(test_img), axis=1),2)
test_img_norm=test_img_norm[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)

test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
test_img_other_input=np.expand_dims(test_img_other_norm, 0)

#Predict and threshold for values above 0.5 probability
#Change the probability threshold to low value (e.g. 0.05) for watershed demo.
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.5).astype(np.uint8)


#plot the result
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask, cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.subplot(234)
plt.title('External Image')
plt.imshow(test_img_other, cmap='gray')
plt.subplot(236)
plt.title('Prediction of external Image')
plt.imshow(prediction_other, cmap='gray')
plt.subplot(235)
plt.title('ground truth')
plt.imshow(test_mask_other, cmap='gray')
plt.show()
