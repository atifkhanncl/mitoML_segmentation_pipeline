
from mitoCytoML.utils import *
from mitoCytoML.methods import *

class segmentor:
    """
    The central class segmentor, used to segment image data.
        Parameters:
            mass channel (np.ndarray): 3D array containing data with `uint16` type.
            membranechannel (np.ndarray, optional): 3D array containing
                data with `uint16` type.
        Methods:
            metrics():
                Provides all segmentation methods implemented.
            
    """

    def __init__(self,membrane_channel, mass_channel, red_green_file_path, folded_region_mask_path):
        
        """
         Constructs all the necessary attributes to create an Segmentor.
            Parameters:
                mass channel (np.ndarray): 3D array containing data with `uint16` type.
                membranechannel (np.ndarray, optional): 3D array containing
                    data with `uint16` type.
            Returns:
                None
        """       
        self.membrane_channel = membrane_channel
        self.mass_channel = mass_channel
        self.c_limit_membrane = 0.1
        self.c_limit_mass = 0.4
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
        """
        Provides all image segmentation methods implemented.
            Paremeters:
                None
            Returns:
                self._methods (list): All image segmentation methods
                    provided by segmentor.
        """
        
        return self.methods
    
    def red_green_image_maker(self):
        
        """
        Creates a Red_Green image combining the membrane and mass channels respectively with 5%-95% contrasting applied.
            Paremeters:
                None
            Returns:
                self.red_green_image: an RGB red_green image with contrasting applied.
        """
        
        if self.red_green_image is None:
            if check_file_exist(self.red_green_file_path):
                self.red_green_image = read_image(self.red_green_file_path)
            else:
                self.red_green_image = red_green_image_maker(self.membrane_channel, self.mass_channel)        
        
        return self.red_green_image
    
    
        
    def fiber_segmentator(self):
        """
        Creates a fiber segmentation masks using stardist ML model.
            Paremeters:
                None
            Returns:
                self.fiber_mask: a label mask with unique colours of each fiber.
        """
                
        red_green_img_no_contrasting = red_green_image_maker_without_contrast(self.membrane_channel, self.mass_channel)
        self.fiber_mask, details = stardist_predict(red_green_img_no_contrasting)
        return self.fiber_mask
    
    def folded_region_segmentor(self):
        """
        Creates a folded region segmentation masks using UNET ML model.
            Paremeters:
                None
            Returns:
                self.folded_region_mask: a binary mask with folded regions set to pixel =255.
        """
        
        if self.red_green_image is None:
            self.red_green_image = self.red_green_image_maker()
            
        if self.folded_region_mask is None:
            if check_file_exist(self.folded_region_mask_path):
                self.folded_region_mask= read_image(self.folded_region_mask_path)
            else:
                self.folded_region_mask= process_folded_region_masks(self.red_green_image)
             
        return self.folded_region_mask
    
    def folded_region_remover(self):
        
        """
        Removes folded region from fiber masks.
            Paremeters:
                None
            Returns:
                self.folded_region_removed_fiber_mask: fiber mask with removed folded regions.
        """
        
        if self.folded_region_mask is None:
            self.folded_region_mask = self.folded_region_segmentor()
            
        if self.fiber_mask is None:
            self.fiber_mask = self.fiber_segmentator() 
        
        self.folded_region_removed_fiber_mask = pixels_remover(self.folded_region_mask, self.fiber_mask)
        
        return self.folded_region_removed_fiber_mask
    
    def fiber_analysability_classifier(self):
        
        """
        Creates a Non analysible (NA) fiber masks using CNN-RF ML model.
            Paremeters:
                None
            Returns:
                self.fiber_analysability_mask: a binary mask with NA fibers set to pixel =255.
        """
        if self.fiber_mask is None:
            self.fiber_mask = self.fiber_segmentator() 
        if self.red_green_image is None:
            self.red_green_image = self.red_green_image_maker()
        
        self.fiber_analysability_mask = classifier_predict(self.red_green_image,self.fiber_mask)
        
        return self.fiber_analysability_mask
    
    
    def non_analysable_fibers_remover(self):
        
        """
        Removes NA fibers from folded regions removed fiber masks .
            Paremeters:
                None
            Returns:
                self.NA_FR_removed_fiber_mask: final fiber mask with both folded regions and NA fibers removed .
        """
        if self.folded_region_removed_fiber_mask is None:
            self.folded_region_removed_fiber_mask = self.folded_region_remover() 
        if self.fiber_analysability_mask is None:
            self.fiber_analysability_mask = self.fiber_analysability_classifier() 
        
        self.NA_FR_removed_fiber_mask = pixels_remover(self.fiber_analysability_mask, self.folded_region_removed_fiber_mask)
        return self.NA_FR_removed_fiber_mask
    
        
    
        
        