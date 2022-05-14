
from model import model   
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


image_directory = '' # ---
mask_directory = '' #-----


SIZE = 128
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'png'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name,0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

#
masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(mask_directory+image_name,0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))


#Normalize images
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.2, random_state = 0)

#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()

###############################################################
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

def get_model():
    return model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()

# save best weights
checkpoint_filepath = 'weights/weights.hdf5' #-------
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

#early stop 
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,verbose=1)

# If starting with pre-trained weights. 
# model.load_weights('weights/New_weight_300.hdf5')

history = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=200, #----- 
                    validation_data=(X_test, y_test), 
                    shuffle=False,
                    callbacks=[model_checkpoint_callback])

# # model.save('last_weights.hdf5')

# ###########################################################
# Evaluate the model
# evaluate model
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")


# # plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#################################
# IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#######################################################################



# #Predict on a few images
# model = get_model()
# model.load_weights('weights/(128*35)images_without_overlap.hdf5') 
# # # 	# evaluate model
# _, acc = model.evaluate(X_test, y_test)
# print("Accuracy = ", (acc * 100.0), "%")

# # # IOU
# y_pred=model.predict(X_test)
# y_pred_thresholded = y_pred > 0.5

# intersection = np.logical_and(y_test, y_pred_thresholded)
# union = np.logical_or(y_test, y_pred_thresholded)
# iou_score = np.sum(intersection) / np.sum(union)
# print("IoU socre is: ", iou_score)
# # #--------------------------------------------
# # # test_img = cv2.imread('/Users/osama-mac/Desktop/master/sim/Testing/testing_patches/img/5155522_01.png', 0)
# # # test_mask = cv2.imread('/Users/osama-mac/Desktop/master/sim/Testing/testing_patches/masks/5155522_01.png', 0)

# # # test_img_other = cv2.imread('/Users/osama-mac/Desktop/master/sim/Testing/testing_patches/img/5155522_00.png', 0)
# # # test_mask_other = cv2.imread('/Users/osama-mac/Desktop/master/sim/Testing/testing_patches/masks/5155522_00.png', 0)

# # # test_img_norm = np.expand_dims(normalize(np.array(test_img), axis=1),2)
# # # test_img_norm=test_img_norm[:,:,0][:,:,None]
# # # test_img_input=np.expand_dims(test_img_norm, 0)

# # # test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
# # # test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
# # # test_img_other_input=np.expand_dims(test_img_other_norm, 0)

# # # #Predict and threshold for values above 0.5 probability
# # # #Change the probability threshold to low value (e.g. 0.05) for watershed demo.
# # # prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
# # # prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.5).astype(np.uint8)


# # # #plot the result
# # # plt.figure(figsize=(16, 8))
# # # plt.subplot(231)
# # # plt.title('Testing Image')
# # # plt.imshow(test_img, cmap='gray')
# # # plt.subplot(232)
# # # plt.title('Testing Label')
# # # plt.imshow(test_mask, cmap='gray')
# # # plt.subplot(233)
# # # plt.title('Prediction on test image')
# # # plt.imshow(prediction, cmap='gray')
# # # plt.subplot(234)
# # # plt.title('External Image')
# # # plt.imshow(test_img_other, cmap='gray')
# # # plt.subplot(236)
# # # plt.title('Prediction of external Image')
# # # plt.imshow(prediction_other, cmap='gray')
# # # plt.subplot(235)
# # # plt.title('ground truth')
# # # plt.imshow(test_mask_other, cmap='gray')
# # # plt.show()


# # #-----------------------------------------------

# images=[s for s in os.listdir('/Users/osama-mac/Desktop/master/sim/Testing/testing_patches/img_128') if not (s.startswith('.'))]
# # images=[(s.split('.png'))[0] for s in images ]

# # # run the predicition

# for img in images:
#     im=cv2.imread('/Users/osama-mac/Desktop/master/sim/Testing/testing_patches/img_128/'+img,0)
#     # msk=cv2.imread('/Users/osama-mac/Desktop/master/sim/Testing/testing_patches/masks/'+img+'.png',0)
#     test_img_norm = np.expand_dims(normalize(np.array(im), axis=1),2)
#     test_img_norm=test_img_norm[:,:,0][:,:,None]
#     test_img_input=np.expand_dims(test_img_norm, 0)
#     prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
#     sav=os.path.join('/Users/osama-mac/Desktop/master/sim/Testing/Segmented_(128*35)_without_overlap',img)
#     plt.imsave(sav,prediction,cmap='gray')

    
    # plt.figure(figsize=(16, 8))
    # plt.subplot(131)
    # plt.title('Testing Image')
    # plt.imshow(im, cmap='gray')
    # plt.subplot(132)
    # plt.title('Ground Truth')
    # plt.imshow(msk, cmap='gray')
    # plt.subplot(133)
    # plt.title('Predicted image')
    # plt.imshow(prediction, cmap='gray')
    # plt.savefig(sav)
    

















