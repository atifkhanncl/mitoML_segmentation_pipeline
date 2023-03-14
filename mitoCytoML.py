from mitoCytoML import segmentor
from mitoCytoML import data
import os
import cv2
data_obj = data.data_obj ()
data_obj.create_dir_structure()
value = input("Please place required raw images into the respective folder then press 'y'")
if value == 'y' or 'Y':
    membrane_list, mass_list = data_obj.get_image_list()
    for infile in membrane_list:
        infile_mass_channel_path= os.path.join(data_obj.mass_channel_dir, infile)
        infile_membrane_channel_path= os.path.join(data_obj.membrane_channel_dir, infile)
        outfile_path_red_green= os.path.join(data_obj.red_green_images_dir, infile)
        outfile_path_raw_fiber_segmentation= os.path.join(data_obj.raw_segmented_masks_dir, infile)
        outfile_path_folded_region= os.path.join(data_obj.FR_masks_dir, infile)
        outfile_path_FR_removed_mask= os.path.join(data_obj.FR_removed_dir, infile)
        outfile_path_NA_mask= os.path.join(data_obj.NA_fibers_dir, infile)
        outfile_path_final_segmented_masks= os.path.join(data_obj.final_segmented_masks_dir, infile)
        
        mass_img= cv2.imread(infile_mass_channel_path,2)
        membrane_img= cv2.imread(infile_membrane_channel_path,2)
        
        segmentor_obj = segmentor.segmentor(membrane_img, mass_img,outfile_path_red_green,outfile_path_folded_region)
        
        red_green_image = segmentor_obj.red_green_image_maker()   
        cv2.imwrite(outfile_path_red_green,red_green_image) 
        
        fiber_mask = segmentor_obj.fiber_segmentator()
        cv2.imwrite(outfile_path_raw_fiber_segmentation,fiber_mask)    
        
        folded_region = segmentor_obj.folded_region_segmentor()
        cv2.imwrite(outfile_path_folded_region ,folded_region)
        
        folded_region_removed_mask = segmentor_obj.folded_region_remover()
        cv2.imwrite(outfile_path_FR_removed_mask ,folded_region_removed_mask)
        
        NA_fiber_mask = segmentor_obj.fiber_analysability_classifier()
        cv2.imwrite(outfile_path_NA_mask ,NA_fiber_mask)
        
        final_segmented_masks = segmentor_obj.non_analysable_fibers_remover()
        cv2.imwrite(outfile_path_final_segmented_masks ,final_segmented_masks)
        
print('All segmentations complete, please check the respective folders')
    