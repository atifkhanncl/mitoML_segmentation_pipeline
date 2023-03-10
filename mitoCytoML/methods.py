from stardist.models import StarDist2D
from csbdeep.utils import Path, normalize
from tensorflow import keras
from keras_unet_collection import models, losses
from keras.models import load_model
import numpy as np
import cv2
from patchify import patchify, unpatchify
import math


def get_stardist_model(model_name='stardist_model',directory= './model_fiber_segmentation/'):
    model = StarDist2D(None, name=model_name, basedir=directory)
    return model
def stardist_predict(img):
    model = get_stardist_model()
    axis_norm = (0,1)
    img_norm = normalize(img, 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img_norm)
    return labels, details
def stardist_predict_with_class(img):
    model = get_stardist_model()
    axis_norm = (0,1)
    img_norm = normalize(img, 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img_norm)
    return labels, details


def get_unet_model(name='FR_model.hdf5',directory= './model_FR/'):
    model = keras.models.load_model(directory+name , custom_objects={"dice_coef": losses.dice_coef })
    return model

def unet_predict(image, patch_size=512):
    step_size= math.gcd(patch_size, image.shape[0],image.shape[1])
    
    model = get_unet_model()
    patches = patchify(image, (patch_size, patch_size,3), step=step_size) 
    predicted_patches = []
    print(step_size)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = np.squeeze(patches[i,j,:,:,:])
            #single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            print(single_patch.shape)
            
            
            single_patch_norm = np.array(single_patch)/255
            single_patch_input=np.expand_dims(single_patch_norm, 0)
            
            single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
            predicted_patches.append(single_patch_prediction)
            
    predicted_patches = np.array(predicted_patches)
    print(predicted_patches.shape)
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], patch_size,patch_size) )
    segm_mask = unpatchify(predicted_patches_reshaped, (image.shape[0],image.shape[1]))
    
    return segm_mask

def get_classifier_model(name='classification_model',directory= './model_fiber_classification/'):
    model = keras.models.load_model(directory+name)
    return model
def classifier_predict(ind_fiber_array):
    ind_fiber_array_scaled = ind_fiber_array/255.0
    
    
    model = get_classifier_model()
    input_img= np.expand_dims(ind_fiber_array_scaled, axis=0)
    prediction = model.predict(input_img)
    prediction_NN = np.argmax(prediction, axis=-1)
    
    
    return prediction_NN
    