from PIL import Image
import PIL.ImageOps  
import numpy as np
import os
import cv2
import csv
from mitoCytoML.utils import dir_create, images_list

raw_images_dir = "./raw_images/"
mass_channel_dir= raw_images_dir+'/mass_channel_images/'
membrane_channel_dir= raw_images_dir+'/membrane_channel_images/'
FR_removed_dir= "./FR_removed_RG_images/"
raw_segmented_masks_dir ="./raw_segmented_masks/"
individual_fibers_dir ='./individual_fibers/'
final_segmented_masks_dir= "./final_segmented_masks/"
model_FR_dir = "./model_FR/"
FR_model_name='FR_model'
model_segmentation_dir= "./model_fiber_segmentation/"
model_classification_dir= "./model_fiber_classification/"


class data_obj:
   

    def __init__(self):
       
        
        self.methods = ['create_dir_structure','get_data_path_list']
    
    def methods(self):
        return self.methods
                
    
    def create_dir_structure(self):
        
        dir_create(raw_images_dir)
        dir_create(mass_channel_dir)
        dir_create(membrane_channel_dir)
        dir_create(FR_removed_dir)
        dir_create(raw_segmented_masks_dir)
        dir_create(individual_fibers_dir)
        dir_create(final_segmented_masks_dir)
        dir_create(model_FR_dir)
        dir_create(FR_model_name)
        dir_create(model_segmentation_dir)
        dir_create(model_classification_dir)
        
        
    def get_data_path_list(self):
        
        img_list_membrane = images_list(membrane_channel_dir)
        img_list_mass = images_list(mass_channel_dir)
        
        assert len(img_list_membrane) != 0 
        assert len(img_list_mass) != 0 
        assert len(img_list_membrane) == len(img_list_mass)
        assert sorted(img_list_membrane) == sorted(img_list_mass)
        
        return img_list_membrane, img_list_mass
        
        
    
    