from mitoCytoML.utils import *
from mitoCytoML.utils import *
from stardist.models import StarDist2D
from cellpose import models as cellpose_models
from stardist import fill_label_holes
from csbdeep.utils import Path, normalize, axes_check_and_normalize
from tensorflow import keras
from keras_unet_collection import models, losses
from keras.models import load_model
import numpy as np
import cv2
import imutils



def get_stardist_model(model_name='stardist_model',directory= './models/model_fiber_segmentation/'):
    model = StarDist2D(None, name=model_name, basedir=directory)
    return model
def get_cellpose_model(model_path='./models/model_fiber_segmentation/cellpose_model/cellpose_model', use_gpu= False):
    model = cellpose_models.CellposeModel(gpu=use_gpu, 
                             pretrained_model=model_path)
    return model

def stardist_predict(img_BGR):
    img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)   
    model = get_stardist_model()
    axis_norm = (0,1)
    img_norm = normalize(img, 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img_norm)
    return labels, details
def stardist_predict_with_class(img_BGR):
    img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    model = get_stardist_model()
    axis_norm = (0,1)
    img_norm = normalize(img, 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img_norm)
    return labels, details
def cellpose_channel_arrangement(Channel_to_use_for_segmentation,Second_segmentation_channel):
    if Channel_to_use_for_segmentation == "Grayscale":
        chan = 0
    elif Channel_to_use_for_segmentation == "Blue":
        chan = 3
    elif Channel_to_use_for_segmentation == "Green":
        chan = 2
    elif Channel_to_use_for_segmentation == "Red":
        chan = 1


    if Second_segmentation_channel == "Blue":
        chan2 = 3
    elif Second_segmentation_channel == "Green":
        chan2 = 2
    elif Second_segmentation_channel == "Red":
        chan2 = 1
    elif Second_segmentation_channel == "None":
        chan2 = 0
    
    return chan, chan2

def cellpose_predict(img_BGR, primary_channel='Green', secondary_channel='Red', fiber_diameter=65):
    img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)    
    chan, chan2= cellpose_channel_arrangement(primary_channel,secondary_channel)   
    model= get_cellpose_model()
    mask, flow, style = model.eval(img, 
                                  channels=[chan, chan2],
                                  diameter=fiber_diameter
                                  
                                  )
    return mask
    
    

def get_unet_model(name='FR_model.hdf5',directory= '.models/model_FR/'):
    model = keras.models.load_model(directory+name , custom_objects={"dice_coef": losses.dice_coef })
    return model

def unet_predict(img, patch_size=512):
    model = get_unet_model()
    image = img.astype(np.uint8) 
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

def process_folded_region_masks(img, contour_area_low_limit=18000):
    folded_regions_mask_= unet_predict(img)
    folded_regions_mask= fill_label_holes(folded_regions_mask_)
    cnts = cv2.findContours(folded_regions_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cnt_area = w * h
        if cnt_area < contour_area_low_limit:
            folded_regions_mask[y:y + h, x:x + w] = 0
    return folded_regions_mask



def NTM_identification_logic(fiber_area,fiber_on_border,fiber_convexity,fiber_lengthSquardedByArea,fiber_aspect_ratio):
    return_flag= True
    if fiber_area>210 & fiber_area<27000 & fiber_on_border== False & fiber_convexity<1.19 & fiber_lengthSquardedByArea < 3.10:
        return_flag= False
        
    return return_flag
        

def get_NTM_mask(instance_mask):
    
    mask= np.copy(instance_mask)
    
    NTM_fibre_ids=[]
    fiber_areas_with_ids= fiber_areas(mask)
    list_of_fiber_ids= list(fiber_areas_with_ids.keys())
    
    for fiber_id in list_of_fiber_ids:
        individual_fiber_mask= np.array(((mask == fiber_id)*255),dtype=np.uint8) 
        fib_area= fiber_areas_with_ids[fiber_id]
        fib_on_border= fiber_at_border(individual_fiber_mask)
        fib_convexity= fiber_convexity(individual_fiber_mask, fib_area)
        fib_aspect_ratio= fiber_aspect_ratio(individual_fiber_mask)
        fib_lengthSquardedByArea= fiber_lengthSquaredbyArea_bbox(individual_fiber_mask, fib_area)
        
        if NTM_identification_logic(fib_area,fib_on_border,fib_convexity,fib_lengthSquardedByArea,fib_aspect_ratio):
            NTM_fibre_ids.append(fiber_id)
    NTM_mask= mask_selected_fibers_array_retain (mask, np.array(NTM_fibre_id))
    #Non_NTM_mask= mask_selected_fibers_array_remove (mask, NTM_fibre_ids.to_numpy())
    
    return NTM_mask
    


def get_FAM_classifier(name='FAM_classifier',directory= './models/model_FAM_classification/'):
    model = keras.models.load_model(directory+name)
    
    return model
def get_FAM_mask(img, fiber_mask):
    fib_mask= np.copy(fiber_mask)
    ind_fiber_list, unique_colours= ind_fiber_image_maker(img, fib_mask)
    model = get_FAM_classifier()
    FAM_ids=[]
    
    #NA_mask= np.zeros(fib_mask.shape,dtype=np.uint8)
    #NA_mask_255 = np.zeros(fib_mask.shape,dtype=np.uint8)
    for img, unique_colour in zip(ind_fiber_list, unique_colours):
        img = cv2.resize(img,dsize=(128,128), interpolation=cv2.INTER_CUBIC)
        img_scaled = img/255.0
        input_img= np.expand_dims(img_scaled, axis=0)
        prediction = np.argmax( model.predict(input_img))
        if prediction > 0:
            FAM_ids.append(unique_colour)
    
    FAM_mask= mask_selected_fibers_array_retain (fib_mask, np.array(FAM_ids))
            
    
    return FAM_mask
    