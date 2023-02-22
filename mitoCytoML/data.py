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
model_segmentation_dir= "./model_fiber_segmentation/"
model_classification_dir= "./model_fiber_classification/"


class data_obj:
   

    def __init__(self):
       
        self.raw_images_dir = raw_images_dir
        self.mass_channel_dir = mass_channel_dir
        self.membrane_channel_dir = membrane_channel_dir
        self.FR_removed_dir = FR_removed_dir
        self.raw_segmented_masks_dir = raw_segmented_masks_dir
        self.individual_fibers_dir = individual_fibers_dir
        self.final_segmented_masks_dir = final_segmented_masks_dir
        self.model_FR_dir = model_FR_dir
        self.model_segmentation_dir =model_segmentation_dir
        self.model_classification_dir = model_classification_dir
        self.methods = ['create_dir_structure','get_image_list', 'get_segmentation_model_name', 'get_FR_model_name', 'get_NASM_classifier_name']
        self.segmentation_model_name = None
        self.FR_model_name = None
        self.NASM_classifier_name = None
        
    def methods(self):
        return self.methods
    
    def raw_images_dir(self):
        return self.raw_images_dir
    
    def mass_channel_dir(self):
        return self.mass_channel_dir
    
    def membrane_channel_dir(self):
        return self.membrane_channel_dir   
    
    def FR_removed_dir(self):
        return self.FR_removed_dir

    def raw_segmented_masks_dir(self):
        return self.raw_segmented_masks_dir
    
    def individual_fibers_dir(self):
        return self.individual_fibers_dir
    
    def final_segmented_masks_dir(self):
        return self.final_segmented_masks_dir
    
    def model_FR_dir(self):
        return self.model_FR_dir
    
    def model_segmentation_dir(self):
        return self.model_segmentation_dir
    
    def model_classification_dir(self):
        return self.model_classification_dir
     
       
    def create_dir_structure(self):
        
        dir_create(raw_images_dir)
        dir_create(mass_channel_dir)
        dir_create(membrane_channel_dir)
        dir_create(FR_removed_dir)
        dir_create(raw_segmented_masks_dir)
        dir_create(individual_fibers_dir)
        dir_create(final_segmented_masks_dir)
        dir_create(model_FR_dir)
        dir_create(model_segmentation_dir)
        dir_create(model_classification_dir)
        
        
    def get_image_list(self):
        
        img_list_membrane = images_list(membrane_channel_dir)
        img_list_mass = images_list(mass_channel_dir)
               
        assert len(img_list_membrane) != 0 
        assert len(img_list_mass) != 0 
        assert len(img_list_membrane) == len(img_list_mass)
        assert sorted(img_list_membrane) == sorted(img_list_mass)
        
        return img_list_membrane, img_list_mass
        
    def get_segmentation_model_name(self):
        
        if self.segmentation_model_name is None:
            self.segmentation_model_name = images_list(self.model_segmentation_dir)
        return self.segmentation_model_name
    
    def get_FR_model_name(self):
        
        if self.FR_model_name is None:
            self.FR_model_name = images_list(self.model_FR_dir)
        return self.FR_model_name
    
    def get_NASM_classifier_name(self):
        
        if self.NASM_classifier_name is None:
            self.NASM_classifier_name = images_list(self.model_classification_dir)
        return self.NASM_classifier_name
        
    
    