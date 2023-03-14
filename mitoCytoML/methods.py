from mitoCytoML.utils import *
from stardist.models import StarDist2D
from csbdeep.utils import Path, normalize
from tensorflow import keras
from keras_unet_collection import models, losses
from keras.models import load_model
import numpy as np
import cv2



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
    model = get_unet_model()
    
    segm_img = np.zeros(image.shape[:2])  #Array with zeros to be filled with segmented values
  
    for i in range(0, image.shape[0], patch_size):   
        for j in range(0, image.shape[1], patch_size):  
            
            if i <= image.shape[0]-patch_size and j <= image.shape[1]-patch_size:
                single_patch = image[i:i+patch_size, j:j+patch_size]
                single_patch_norm = np.array(single_patch)/255
                single_patch_shape = single_patch_norm.shape[:2]
                single_patch_input = np.expand_dims(single_patch_norm, 0)
                single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
                segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
                
               
            if i > image.shape[0]-patch_size and j <= image.shape[1]-patch_size:
                single_patch = image[image.shape[0]-patch_size:image.shape[0], j:j+patch_size]
                single_patch_norm = np.array(single_patch)/255
                single_patch_shape = single_patch_norm.shape[:2]
                single_patch_input = np.expand_dims(single_patch_norm, 0)
                single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
                segm_img[image.shape[0]-patch_size:image.shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
                
            if i <= image.shape[0]-patch_size and j > image.shape[1]-patch_size:
                single_patch = image[i:i+patch_size, image.shape[1]-patch_size:image.shape[1]]
                single_patch_norm = np.array(single_patch)/255
                single_patch_shape = single_patch_norm.shape[:2]
                single_patch_input = np.expand_dims(single_patch_norm, 0)
                single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
                segm_img[i:i+single_patch_shape[0], image.shape[1]-patch_size:image.shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
                
            if i > image.shape[0]-patch_size and j > image.shape[1]-patch_size:
                single_patch = image[image.shape[0]-patch_size:image.shape[0], image.shape[1]-patch_size:image.shape[1]]
                single_patch_norm = np.array(single_patch)/255
                single_patch_shape = single_patch_norm.shape[:2]
                single_patch_input = np.expand_dims(single_patch_norm, 0)
                single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
                segm_img[image.shape[0]-patch_size:image.shape[0], image.shape[1]-patch_size:image.shape[1]] += cv2.resize(single_patch_prediction,
                                                                                                                           single_patch_shape[::-1])
               
    segm_mask= np.zeros((segm_img.shape[0], segm_img.shape[1]), dtype=np.uint8)           
    segm_mask[segm_img > 0 ] = 255       
   
    
    return segm_mask

def get_classifier_model(name='classification_model',directory= './model_fiber_classification/'):
    model = keras.models.load_model(directory+name)
    
    return model
def classifier_predict(img, fib_mask):
    
    ind_fiber_list, unique_colours= ind_fiber_image_maker(img, fib_mask)
    model = get_classifier_model()
    
    NA_mask= np.zeros(fib_mask.shape,dtype=np.uint8)
    NA_mask_255 = np.zeros(fib_mask.shape,dtype=np.uint8)
    for img, unique_colour in zip(ind_fiber_list, unique_colours):
        img = cv2.resize(img,dsize=(128,128), interpolation=cv2.INTER_CUBIC)
        img_scaled = img/255.0
        input_img= np.expand_dims(img_scaled, axis=0)
        prediction = np.argmax( model.predict(input_img))
        NA_mask[fib_mask == unique_colour]= prediction
    NA_mask_255[NA_mask > 0] == 255
    
    return NA_mask_255
    