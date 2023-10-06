
from mitoCytoML.utils import *
from mitoCytoML.methods import *

class segmentor:
    """
    The central class segmentor, used to segment image data.
        Parameters:
            mass channel (np.ndarray): 3D array containing data with `uint16` type.
            membrane channel (np.ndarray, optional): 3D array containing
                data with `uint16` type.
        Methods:
            metrics():
                Provides all segmentation methods implemented.
            
    """

    def __init__(self,membrane_channel, mass_channel,manual_mask, red_green_file_path, fiber_mask_path,stardist_mask_path,cellpose_mask_path, NTM_mask_path, FAM_mask_path, FR_mask_path, final_mask_path):
        
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
        self.manual_mask= manual_mask
        self.c_limit_membrane = 0.1
        self.c_limit_mass = 0.4
        self.red_green_file_path = red_green_file_path
        self.fiber_mask_path = fiber_mask_path
        self.stardist_mask_path= stardist_mask_path
        self.cellpose_mask_path=cellpose_mask_path
        self.NTM_mask_path = NTM_mask_path
        self.FAM_mask_path = FAM_mask_path       
        self.FR_mask_path = FR_mask_path
        self.final_mask_path = final_mask_path
        self.methods = ['red_green_image_maker','fiber_segmentator','NTM_mask_maker','NTM_remover','FAM_mask_maker','FAM_remover',
                         'folded_region_segmentor', 'folded_region_remover',
                        'non_analysable_fibers_remover', 'quality_metrics_calculator']
        self.red_green_image = None
        self.fiber_mask = None
        self.fiber_mask_stardist = None
        self.fiber_mask_cellpose = None
        self.NTM_mask = None
        self.NTM_removed_mask = None
        self.FAM_mask = None
        self.FAM_removed_mask = None
        self.folded_region_mask = None
        self.folded_region_removed_fiber_mask = None
        self.curated_fiber_mask = None
        self.fiber_metrics = None
        self.quality_metrics = None
        
        

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
                
        if self.red_green_image is None:
            self.red_green_image = self.red_green_image_maker()
            
        if self.fiber_mask is None:
            if check_file_exist(self.fiber_mask_path):
                self.fiber_mask= read_image(self.fiber_mask_path)
                self.fiber_mask_stardist= read_image(self.stardist_mask_path)
                self.fiber_mask_cellpose= read_image(self.cellpose_mask_path)
                
                
            else:
                self.fiber_mask_stardist, details = stardist_predict(self.red_green_image)
                self.fiber_mask_cellpose = cellpose_predict(self.red_green_image)
                self.fiber_mask = self.fiber_mask_cellpose
                #segmentor_model_to_use= input("By default the pipeline uses cellpose to segment myofibres if this ok press enter, otherwise to use stardist press 'S' and enter")
                #if segmentor_model_to_use.casefold() == 'S' :
                #    self.fiber_mask= self.fiber_mask_stardist
                #else:
                #    self.fiber_mask= self.fiber_mask_cellpose
            
        return self.fiber_mask, self.fiber_mask_stardist, self.fiber_mask_cellpose
    
    
    def NTM_mask_maker(self):
        
        """
        Creates an instance mask of Non-Transverse Myofibres (NTM) using thersholds.
            Paremeters:
                None
            Returns:
                self.NTM_mask: an instance mask of NT myofibres.
        """
        if self.fiber_mask is None:
            self.fiber_mask = self.fiber_segmentator() 
        if self.NTM_mask is None:
            if check_file_exist(self.NTM_mask_path):
                self.NTM_mask= read_image(self.NTM_mask_path)
            else:
                self.NTM_mask, self.fiber_metrics = get_NTM_mask(self.fiber_mask)
        
        return self.NTM_mask, self.fiber_metrics
    
    
    def NTM_remover(self):
        
        
        if self.fiber_mask is None:
            self.fiber_mask = self.fiber_segmentator() 
        if self.NTM_mask is None:
            self.NTM_mask, self.fiber_metrics = self.NTM_mask_maker() 
        self.NTM_removed_mask= from_mask_to_remove_mask (self.fiber_mask,self.NTM_mask)
        
        
        return self.NTM_removed_mask
    
    
    def FAM_mask_maker(self):
        """
        Creates an instance mask of Non-Transverse Myofibres (NTM) using thersholds.
            Paremeters:
                None
            Returns:
                self.NTM_mask: an instance mask of NT myofibres.
        """
        if self.red_green_image is None:
            self.red_green_image = self.red_green_image_maker()
        if self.fiber_mask is None:
            self.fiber_mask = self.fiber_segmentator() 
        if self.FAM_mask is None:
            if check_file_exist(self.FAM_mask_path):
                self.FAM_mask= read_image(self.FAM_mask_path)
            else:
                self.FAM_mask = get_FAM_mask(self.red_green_image, self.fiber_mask)
        
        
        
        return self.FAM_mask
    
    
    def FAM_remover(self):
        
        if self.fiber_mask is None:
            self.fiber_mask = self.fiber_segmentator() 
        if self.FAM_mask is None:
            self.FAM_mask = self.FAM_mask_maker() 
        self.FAM_removed_mask= from_mask_to_remove_mask (self.fiber_mask,self.FAM_mask)
        
        
        return self.FAM_removed_mask
    
    
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
            if check_file_exist(self.FR_mask_path):
                self.folded_region_mask= read_image(self.FR_mask_path)
            else:
                #self.folded_region_mask= process_folded_region_masks(self.red_green_image)
                self.folded_region_mask=give_blank_mask(self.red_green_image)
             
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
        
        self.folded_region_removed_fiber_mask = from_mask_to_remove_mask( self.fiber_mask, self.folded_region_mask)
        
        return self.folded_region_removed_fiber_mask
    
    
   
    
    
    
    
    def non_analysable_fibers_remover(self):
        
        """
        Removes NA fibers from folded regions removed fiber masks .
            Paremeters:
                None
            Returns:
                self.NA_FR_removed_fiber_mask: final fiber mask with both folded regions and NA fibers removed .
        """
     
        if self.fiber_mask is None:
            self.fiber_mask = self.fiber_segmentator() 
        if self.NTM_mask is None:
            self.NTM_mask = self.NTM_mask_maker()
        if self.FAM_mask is None:
            self.FAM_mask = self.FAM_mask_maker() 
        if self.folded_region_mask is None:
            self.folded_region_mask = self.folded_region_segmentor()
        remove_NTM= from_mask_to_remove_mask (self.fiber_mask,self.NTM_mask)
        remove_FAM_NTM= from_mask_to_remove_mask (remove_NTM,self.FAM_mask)
        self.curated_fiber_mask = from_mask_to_remove_mask (remove_FAM_NTM,self.folded_region_mask)
        return self.curated_fiber_mask
    
    
    def quality_metrics_calculator(self):
        
        """
        Calculates quality metrics between manual annotation and prediction .
            Paremeters:
                None
            Returns:
                self.quality_metrics: An nd array of all quality metrics .
        """
        if self.red_green_image is None:
            self.red_green_image = self.red_green_image_maker()     
        if self.fiber_mask is None:
            self.fiber_mask = self.fiber_segmentator() 
        self.quality_metrics= get_quality_metrics(self.red_green_image,self.manual_mask, self.fiber_mask, element_size=2 )
       
        return self.quality_metrics
    
        
    
        
        