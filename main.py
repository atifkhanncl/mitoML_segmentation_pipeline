from mitoCytoML import segmentor
from mitoCytoML import data
import os
import cv2
import numpy as np
data_obj = data.data_obj ()
data_obj.create_dir_structure()

value = input("Please place required raw images into the respective folder then press 'y' and enter")
if value == 'y' or 'Y':
    membrane_list, mass_list = data_obj.get_image_list()
    for infile in membrane_list:
        infile_mass_channel_path= os.path.join(data_obj.mass_channel_dir, infile)
        infile_membrane_channel_path= os.path.join(data_obj.membrane_channel_dir, infile)
        infile_manual_sample_mask_path= os.path.join(data_obj.manual_sample_masks_dir, infile)
        
        outfile_path_red_green= os.path.join(data_obj.red_green_images_dir, infile)
        
        outfile_path_raw_fiber_segmentation= os.path.join(data_obj.raw_segmented_masks_dir, infile)        
        outfile_path_raw_segmented_masks_stardist= os.path.join(data_obj.raw_segmented_masks_stardist_dir, infile)
        outfile_path_raw_segmented_masks_cellpose= os.path.join(data_obj.raw_segmented_masks_cellpose_dir, infile) 
        
        outfile_path_NTM_mask= os.path.join(data_obj.NTM_masks_dir, infile)
        outfile_path_NTM_removed_mask= os.path.join(data_obj.NTM_removed_masks_dir, infile)
        
        outfile_path_FAM_mask= os.path.join(data_obj.FAM_masks_dir, infile)
        outfile_path_FAM_removed_mask= os.path.join(data_obj.FAM_removed_masks_dir, infile)
        
        outfile_path_FR_mask= os.path.join(data_obj.FR_masks_dir, infile)
        outfile_path_FR_removed_mask= os.path.join(data_obj.FR_removed_dir, infile)
        
        outfile_path_final_segmented_mask= os.path.join(data_obj.final_segmented_masks_dir, infile)
        
        outfile_path_metrics= os.path.join(data_obj.metrics_dir, infile.split('.')[0]+'.csv')
        
        mass_img= cv2.imread(infile_mass_channel_path,-1)
        membrane_img= cv2.imread(infile_membrane_channel_path,-1)
        manual_mask= cv2.imread(infile_manual_sample_mask_path,-1)
        
        segmentor_obj = segmentor.segmentor(membrane_img, mass_img,outfile_path_red_green,outfile_path_raw_fiber_segmentation,outfile_path_NTM_mask,outfile_path_FAM_mask,outfile_path_FR_mask,outfile_path_final_segmented_mask, outfile_path_metrics )
        
        red_green_image = segmentor_obj.red_green_image_maker()   
        cv2.imwrite(outfile_path_red_green,red_green_image) 
        
        fiber_mask,fiber_mask_stardist, fiber_mask_cellpose  = segmentor_obj.fiber_segmentator()
        cv2.imwrite(outfile_path_raw_fiber_segmentation,fiber_mask)
        cv2.imwrite(outfile_path_raw_segmented_masks_stardist,fiber_mask_stardist)
        cv2.imwrite(outfile_path_raw_segmented_masks_cellpose,fiber_mask_cellpose)
        
          
        NTM_mask = segmentor_obj.NTM_mask_maker()
        cv2.imwrite(outfile_path_NTM_mask ,NTM_mask)
        NTM_removed_mask = segmentor_obj.NTM_remover()
        cv2.imwrite(outfile_path_NTM_removed_mask ,NTM_removed_mask)
        
        FAM_mask = segmentor_obj.FAM_mask_maker()
        cv2.imwrite(outfile_path_FAM_mask ,FAM_mask)
        FAM_removed_mask = segmentor_obj.FAM_remover()
        cv2.imwrite(outfile_path_FAM_removed_mask ,FAM_removed_mask)
        
        FR_mask = segmentor_obj.folded_region_segmentor()
        cv2.imwrite(outfile_path_FR_mask ,FR_mask)
        
        FR_removed_mask = segmentor_obj.folded_region_remover()
        cv2.imwrite(outfile_path_FR_removed_mask ,FR_removed_mask)
      
        
        final_segmented_masks = segmentor_obj.non_analysable_fibers_remover()
        cv2.imwrite(outfile_path_final_segmented_masks ,final_segmented_masks)
        
        if manual_mask is not None:
            metrics_array= segmentor_obj.metrics_calculator()
            np.savetext(outfile_path_metrics,metrics_array, delimiter=",")
            
            
        
print('All segmentations complete, please check the respective folders')
    