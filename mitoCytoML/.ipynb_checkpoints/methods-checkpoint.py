from mitoCytoML.utils import *
from mitoCytoML.utils import *
from stardist.models import StarDist2D
from cellpose import models as cellpose_models
from stardist import fill_label_holes
from csbdeep.utils import Path, normalize, axes_check_and_normalize
from tensorflow import keras
from keras_unet_collection import models, losses
from keras.models import load_model
from keras.applications.vgg16 import VGG16
import xgboost as xgb
import numpy as np
import cv2
import imutils



def get_stardist_model(model_name='stardist_model',directory= './models/model_fiber_segmentation/'):
    model = StarDist2D(None, name=model_name, basedir=directory)
    return model
def get_cellpose_model_imc(model_path='./models/model_fiber_segmentation/cellpose_model/cellpose_model_imc', use_gpu= False):
    model = cellpose_models.CellposeModel(gpu=use_gpu, 
                             pretrained_model=model_path)
    return model
def get_cellpose_model_qif(model_path='./models/model_fiber_segmentation/cellpose_model/cellpose_model_qif', use_gpu= False):
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

def cellpose_predict(img_BGR,is_imc,is_qif,model_match_threshold, primary_channel='Green', secondary_channel='Red',cell_prob_thershold=0.0):
    
    if is_imc:
        fiber_diameter=65
        model= get_cellpose_model_imc()
    if is_qif:
        fiber_diameter=200
        model= get_cellpose_model_qif()
        
    flow_style_threshold = (31.0 - model_match_threshold) / 10.
           
    img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)    
    chan, chan2= cellpose_channel_arrangement(primary_channel,secondary_channel)   
    
    mask, flow, style = model.eval(img, 
                                  channels=[chan, chan2],
                                  diameter=fiber_diameter,
                                  flow_threshold=flow_style_threshold, 
                                  cellprob_threshold=cell_prob_thershold
                                
                                   
                                  
                                  )
    return mask
    
    

def get_unet_model(name='FR_model.hdf5',directory= './models/model_FR/'):
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

   

def NTM_identification_logic(fiber_area,fiber_on_border,fiber_convexity,fiber_lengthSquardedByArea,fiber_aspect_ratio, is_imc, is_qif, qif_resolution=0.323):
    return_flag= True
    if is_imc:
        if fiber_area>210 and fiber_area<27000 and fiber_on_border is False and fiber_convexity<1.19 and fiber_lengthSquardedByArea < 3.10 and (not(fiber_aspect_ratio>2.2 and fiber_lengthSquardedByArea>1.8)):
            return_flag= False
    if is_qif:
        area_resolution=1/(qif_resolution*qif_resolution)
        if fiber_area>(210*area_resolution) and fiber_area<(27000*area_resolution) and fiber_on_border is False and fiber_convexity<1.19 and fiber_lengthSquardedByArea < 3.10 and (not(fiber_aspect_ratio>2.2 and fiber_lengthSquaredByArea>1.8)):
            return_flag=False
        
    return return_flag
        

def get_NTM_mask(instance_mask,is_imc,is_qif):
    
    mask= np.copy(instance_mask)
    
    NTM_fibre_ids=[]
    fiber_areas_with_ids= fiber_areas(mask)
    list_of_fiber_ids= list(fiber_areas_with_ids.keys())
    fiber_metrics=[]
    for fiber_id in list_of_fiber_ids:
        individual_fiber_mask= np.array(((mask == fiber_id)*255),dtype=np.uint8) 
        fib_area= fiber_areas_with_ids[fiber_id]
        fib_on_border= fiber_at_border(individual_fiber_mask)
        fib_convexity= fiber_convexity(individual_fiber_mask, fib_area)
        fib_aspect_ratio= fiber_aspect_ratio(individual_fiber_mask)
        fib_lengthSquardedByArea= fiber_lengthSquaredbyArea_bbox(individual_fiber_mask, fib_area)
        fiber_metrics.append((fib_area,fib_on_border,fib_convexity,fib_lengthSquardedByArea,fib_aspect_ratio))
        
        if NTM_identification_logic(fib_area,fib_on_border,fib_convexity,fib_lengthSquardedByArea,fib_aspect_ratio,is_imc,is_qif):
            NTM_fibre_ids.append(fiber_id)
    NTM_mask= mask_selected_fibers_array_retain (mask, np.array(NTM_fibre_ids))
    #Non_NTM_mask= mask_selected_fibers_array_remove (mask, NTM_fibre_ids.to_numpy())
    
    return NTM_mask,fiber_metrics
    
def get_VGG16_features(scaled_img, SIZE=128):
    VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
    for layer in VGG_model.layers:
        layer.trainable = False
    
    predicted_features= VGG_model.predict(scaled_img)
    predicted_features= predicted_features.reshape(predicted_features.shape[0], -1)
    
    return predicted_features

def get_FAM_classifier(name='xgBoost.json',directory= './models/model_FAM_classification/FAM_classifier/'):
    model = xgb.XGBClassifier()
    model.load_model(directory+name)
    return model
    
    return model
def get_FAM_mask(img, fiber_mask, SIZE=128):
    fib_mask= np.copy(fiber_mask)
    ind_fiber_list, unique_colours= ind_fiber_image_maker(img, fib_mask)
    model = get_FAM_classifier()
    FAM_ids=[]
    
    #NA_mask= np.zeros(fib_mask.shape,dtype=np.uint8)
    #NA_mask_255 = np.zeros(fib_mask.shape,dtype=np.uint8)
    for img, unique_colour in zip(ind_fiber_list, unique_colours):
        img = cv2.resize(img,(SIZE,SIZE))
        img_scaled = img/255.0
        input_img= np.expand_dims(img_scaled, axis=0)
        input_img_features = get_VGG16_features(input_img, SIZE)
        model=  get_FAM_classifier()
        prediction = model.predict(input_img_features)[0]
        
        if prediction == 0:
            FAM_ids.append(unique_colour)
    
    FAM_mask= mask_selected_fibers_array_retain (fib_mask, np.array(FAM_ids))
            
    
    return FAM_mask

def get_quality_metrics(img,mask_gt, mask_to_assess, element_size=2 ): # element_size= 2 for IMC and 6 for IF
    
    contours_GT= contour_finder_instance_mask(mask_gt)
    contours_Assess= contour_finder_instance_mask(mask_to_assess)
    
    gt_memb_vals=[]
    gt_mass_vals=[]
    as_memb_vals=[]
    as_mass_vals=[]
    IoUs=[]
    for i, cont_gt in enumerate(contours_GT):
        
        for j, cont_as in enumerate(contours_Assess):
            
            if are_same_fibres(contours_GT[i],contours_Assess[j]):
                gt_memb_val, gt_mass_val= memb_mass_cal(img,contours_GT[i],element_size)
                gt_memb_vals.append(gt_memb_val)
                gt_mass_vals.append(gt_mass_val)
                
                as_memb_val, as_mass_val= memb_mass_cal(img,contours_Assess[j],element_size)
                as_memb_vals.append(as_memb_val)
                as_mass_vals.append(as_mass_val)
                
                IoU = IoU_two_contours(mask_gt.shape[0:2],contours_GT[i], contours_Assess[j])
                IoUs.append(IoU)
                
                break
    IoUs= np.asarray(IoUs, dtype=np.float16)
    r_memb= np.corrcoef(gt_memb_vals, as_memb_vals)[1,0]
    r_mass= np.corrcoef(gt_mass_vals, as_mass_vals)[1,0]
    
    IoU_mean= np.mean(IoUs)
    
    IoU_accuracy_at_80= (((IoUs >0.8).sum())/len(IoUs))*100
    IoU_accuracy_at_90= (((IoUs >0.9).sum())/len(IoUs))*100
    IoU_accuracy_at_95= (((IoUs >0.95).sum())/len(IoUs))*100
    
    return (r_memb,r_mass, IoU_mean,IoU_accuracy_at_80,IoU_accuracy_at_90,IoU_accuracy_at_95 )
    
    
                
                
                
                
                
    
    