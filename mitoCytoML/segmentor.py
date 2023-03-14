
from mitoCytoML.utils import *
from mitoCytoML.methods import *

class segmentor:
   

    def __init__(self,membrane_channel, mass_channel, red_green_file_path, folded_region_mask_path):
       
        self.membrane_channel = membrane_channel
        self.mass_channel = mass_channel
        self.c_limit_membrane = 0.1
        self.c_limit_mass = 0.1
        self.red_green_file_path = red_green_file_path
        self.folded_region_mask_path = folded_region_mask_path
        self.methods = ['red_green_image_maker','fiber_segmentator',
                         'folded_region_segmentor', 'folded_region_remover',
                        'fiber_analysability_classifier','non_analysable_fibers_remover']
        self.red_green_image = None
        self.fiber_mask = None
        self.folded_region_mask = None
        self.fiber_analysability_mask = None
        self.folded_region_removed_fiber_mask = None
        self.NA_FR_removed_fiber_mask = None

    def methods(self):
        
        return self.methods
    
    def red_green_image_maker(self):
        
        if self.red_green_image is None:
            if check_file_exist(self.red_green_file_path):
                self.red_green_image = read_image(self.red_green_file_path)
            else:
                self.red_green_image = red_green_image_maker(self.membrane_channel, self.mass_channel,self.c_limit_membrane,self.c_limit_mass)        
        
        return self.red_green_image
    
    
    def fiber_segmentator(self):
        
        if self.red_green_image is None:
            self.red_green_image = self.red_green_image_maker()
        
        fiber_mask, details = stardist_predict(self.red_green_image)
        return fiber_mask
    
    def folded_region_segmentor(self):
        
        
        if self.folded_region_mask is None:
            if check_file_exist(self.folded_region_mask_path):
                self.folded_region_mask= read_image(self.folded_region_mask_path)
            else:
                self.folded_region_mask= unet_predict(self.red_green_image)
             
        return self.folded_region_mask
    
    def folded_region_remover(self):
        
        if self.red_green_image is None:
            self.red_green_image = self.red_green_image_maker()
        
        if self.folded_region_mask is None:
            self.folded_region_mask = self.folded_region_segmentor()
            
        if self.fiber_mask is None:
            self.fiber_mask = self.fiber_segmentator() 
        
        self.folded_region_removed_fiber_mask = pixels_remover(self.folded_region_mask, self.fiber_mask)
        
        return self.folded_region_removed_fiber_mask
    
    def fiber_analysability_classifier(self):
        if self.fiber_mask is None:
            self.fiber_mask = self.fiber_segmentator() 
        if self.red_green_image is None:
            self.red_green_image = self.red_green_image_maker()
        
        self.fiber_analysability_mask = classifier_predict(self.red_green_image,self.fiber_mask)
        
        return self.fiber_analysability_mask
    
    
    def non_analysable_fibers_remover(self):
        if self.folded_region_removed_fiber_mask is None:
            self.folded_region_removed_fiber_mask = self.folded_region_remover() 
        if self.fiber_analysability_mask is None:
            self.fiber_analysability_mask = self.fiber_analysability_classifier() 
        
        self.NA_FR_removed_fiber_mask = pixels_remover(self.fiber_analysability_mask, self.folded_region_removed_fiber_mask)
        return self.NA_FR_removed_fiber_mask
    
        
    
        
        