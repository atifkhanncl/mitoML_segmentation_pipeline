from PIL import Image
import PIL.ImageOps  
import numpy as np
import os
import cv2
import csv
from mitoCytoML.utils import dir_create, file_list

raw_images_dir = "./input/raw_images/"
mass_channel_dir= raw_images_dir+'/mass_channel_images/'
membrane_channel_dir= raw_images_dir+'/membrane_channel_images/'

manual_sample_masks_dir= './input/manual_sample_masks/'

red_green_images_dir= "./output/red_green_images"

raw_segmented_masks_dir ="./output/Step_1/raw_segmented_masks/"
#raw_segmented_masks_stardist_dir ="./output/Step_1/raw_segmented_masks/stardist"
#raw_segmented_masks_cellpose_dir ="./output/Step_1/raw_segmented_masks/cellpose"

NTM_masks_dir ='./output/Step_2A/NTM_masks/'
NTM_removed_masks_dir ='./output/Step_2A/NTM_removed_masks/'

FAM_masks_dir ='./output/Step_2B/FAM_masks/'
FAM_removed_masks_dir ='./output/Step_2B/FAM_removed_masks/'

FR_masks_dir= "./output/Step_3/FR_masks"
FR_removed_dir= "./output/Step_3/FR_removed_fiber_mask/"


final_segmented_masks_dir= "./output/Step_4_Final_segmented_masks/"

metrics_dir= "./output/Step_4_Metrics/"


model_segmentation_dir= "./models/model_fiber_segmentation/"
model_FAM_classification_dir= "./models/model_FAM_classification/"
model_FR_dir = "./models/model_FR/"


class data_obj:
    """
    The class data, used to create directory structure.
        Parameters:
            None
        Methods:
            metrics():
                Provides all data methods implemented.
            
    """

    def __init__(self):
        
        """
         Constructs all the necessary attributes to create an data_obj.
            Parameters:
                None
            Returns:
                None
        """
       
        self.raw_images_dir = raw_images_dir
        self.mass_channel_dir = mass_channel_dir
        self.membrane_channel_dir = membrane_channel_dir
        
        self.manual_sample_masks_dir=manual_sample_masks_dir
        
        self.red_green_images_dir = red_green_images_dir
        
        
        self.raw_segmented_masks_dir = raw_segmented_masks_dir
        #self.raw_segmented_masks_stardist_dir = raw_segmented_masks_stardist_dir
        #self.raw_segmented_masks_cellpose_dir = raw_segmented_masks_cellpose_dir
        
        self.NTM_masks_dir= NTM_masks_dir
        self.NTM_removed_masks_dir= NTM_removed_masks_dir
        
        self.FAM_masks_dir= FAM_masks_dir
        self.FAM_removed_masks_dir= FAM_removed_masks_dir
        
        self.FR_masks_dir= FR_masks_dir
        self.FR_removed_dir= FR_removed_dir
        
        self.final_segmented_masks_dir = final_segmented_masks_dir
        
        self.metrics_dir= metrics_dir
        
        self.model_segmentation_dir= model_segmentation_dir
        self.model_FAM_classification_dir= model_FAM_classification_dir
        self.model_FR_dir= model_FR_dir
        
               
        self.methods = ['create_dir_structure','get_image_list', 'get_segmentation_model_name', 'get_FR_model_name', 'get_FAM_classifier_name']
        self.segmentation_model_name = None
        self.FR_model_name = None
        self.FAM_classifier_name = None
        
    def methods(self):
        return self.methods
    
    def raw_images_dir(self):
        return self.raw_images_dir
    
    def mass_channel_dir(self):
        return self.mass_channel_dir
    
    def membrane_channel_dir(self):
        return self.membrane_channel_dir  
    
    def manual_sample_masks_dir(self):
        return self.manual_sample_masks_dir
    
    def red_green_images_dir(self):
        return self.red_green_images_dir
    
    def raw_segmented_masks_dir(self):
        return self.raw_segmented_masks_dir
    
    #def raw_segmented_masks_stardist_dir(self):
        #return self.raw_segmented_masks_stardist_dir
    
    #def raw_segmented_masks_cellpose_dir(self):
        #return self.raw_segmented_masks_cellpose_dir
    
    def NTM_masks_dir(self):
        return NTM_masks_dir
    
    def NTM_removed_masks_dir(self):
        return NTM_removed_masks_dir
    
    def FAM_masks_dir(self):
        return FAM_masks_dir
    
    def FAM_removed_masks_dir(self):
        return FAM_removed_masks_dir
    
    def FR_masks_dir(self):
        return FR_masks_dir
    
    def FR_removed_dir(self):
        return FR_removed_dir
    
    def final_segmented_masks_dir(self):
        return final_segmented_masks_dir
    
    def metrics_dir(self):
        return metrics_dir
    
    def model_segmentation_dir(self):
        return model_segmentation_dir
    
    def model_FAM_classification_dir(self):
        return model_FAM_classification_dir
    
    def model_FR_dir(self):
        return model_FR_dir
    
     
       
    def create_dir_structure(self):
        
        """
        Creates all directories required.
            Paremeters:
                None
            Returns:
                None
        """
        
        dir_create(raw_images_dir)
        dir_create(mass_channel_dir)
        dir_create(membrane_channel_dir)
        
        dir_create(red_green_images_dir)
        
        dir_create(manual_sample_masks_dir)
        
        
        dir_create(raw_segmented_masks_dir)
        #dir_create(raw_segmented_masks_stardist_dir)
        #dir_create(raw_segmented_masks_cellpose_dir)
        
        dir_create(NTM_masks_dir)
        dir_create(NTM_removed_masks_dir)
        
        dir_create(FAM_masks_dir)
        dir_create(FAM_removed_masks_dir)
        
        dir_create(FR_masks_dir)
        dir_create(FR_removed_dir)
        
        dir_create(final_segmented_masks_dir)
        
        dir_create(metrics_dir)
        
        
        dir_create(model_segmentation_dir)
        dir_create(model_FAM_classification_dir)
        dir_create(model_FR_dir)
        
        
    def get_image_list(self):
        
        img_list_membrane = file_list(membrane_channel_dir)
        img_list_mass = file_list(mass_channel_dir)
               
        assert len(img_list_membrane) != 0 
        assert len(img_list_mass) != 0 
        assert len(img_list_membrane) == len(img_list_mass)
        assert sorted(img_list_membrane) == sorted(img_list_mass)
        
        return img_list_membrane, img_list_mass
        
    def get_segmentation_model_name(self):
        
        if self.segmentation_model_name is None:
            self.segmentation_model_name = file_list(self.model_segmentation_dir)
        return self.segmentation_model_name
    
    def get_FR_model_name(self):
        
        if self.FR_model_name is None:
            self.FR_model_name = file_list(self.model_FR_dir)
        return self.FR_model_name
    
    def get_FAM_classifier_name(self):
        
        if self.NASM_classifier_name is None:
            self.NASM_classifier_name = file_list(self.model_classification_dir)
        return self.NASM_classifier_name
        
    
    