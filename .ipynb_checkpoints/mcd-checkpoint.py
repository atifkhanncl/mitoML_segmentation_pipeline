input_folder = "C:\\Users\\Conor\\Documents\\parseMCD\\data"
output_folder = "C:\\Users\\Conor\\Documents\\parseMCD\\output"

masschan = "VDAC1"
membranechan = "DMD"

import os
os.environ['output_folder'] = output_folder
from mitoCytoML import segmentor
from mitoCytoML import data_mcd
import cv2
import numpy as np
from readimc import MCDFile, TXTFile
import glob

data_obj = data_mcd.data_obj ()
data_obj.create_dir_structure() # Should really be passing output directory name as a variable here

mcd_list = glob.glob(os.path.join(input_folder,"*.mcd"))

for mcd in mcd_list:
    mcd_root = os.path.basename(mcd_list[0]).replace(".mcd","")
    with MCDFile(mcd) as f:
        slide = f.slides[0]
        num_acquisitions = len(slide.acquisitions)
        for acquisition_num in range(0,num_acquisitions):
            acquisition = slide.acquisitions[acquisition_num]
            desc = acquisition.metadata["Description"]
            chans = acquisition.channel_labels
            mass_img = f.read_acquisition(acquisition)[[c == masschan for c in chans],:,:][0]
            membrane_img = f.read_acquisition(acquisition)[[c == membranechan for c in chans],:,:][0]
            infile_manual_sample_mask_path= os.path.join(os.path.dirname(mcd), "{}_{}.tiff".format(mcd_root,desc)) # Shouldn't this point to output directory?
            manual_mask= cv2.imread(infile_manual_sample_mask_path,-1)
            if manual_mask is None:
                manual_mask= np.zeros(mass_img.shape[0:2]).astype(np.uint16)

            # Why doesn't all this path generation happen in the data class?
            outfile_path_red_green= os.path.join(data_obj.red_green_images_dir, "{}_{}.tiff".format(mcd_root,desc))
    
            outfile_path_raw_fiber_segmentation= os.path.join(output_folder, data_obj.raw_segmented_masks_dir, "{}_{}.tiff".format(mcd_root,desc))        
            outfile_path_raw_segmented_masks_stardist= os.path.join(data_obj.raw_segmented_masks_stardist_dir, "{}_{}.tiff".format(mcd_root,desc))
            outfile_path_raw_segmented_masks_cellpose= os.path.join(data_obj.raw_segmented_masks_cellpose_dir, "{}_{}.tiff".format(mcd_root,desc)) 
            
            outfile_path_NTM_mask= os.path.join(data_obj.NTM_masks_dir, "{}_{}.tiff".format(mcd_root,desc))
            outfile_path_NTM_removed_mask= os.path.join(data_obj.NTM_removed_masks_dir, "{}_{}.tiff".format(mcd_root,desc))
            
            outfile_path_FAM_mask= os.path.join(data_obj.FAM_masks_dir, "{}_{}.tiff".format(mcd_root,desc))
            outfile_path_FAM_removed_mask= os.path.join(data_obj.FAM_removed_masks_dir, "{}_{}.tiff".format(mcd_root,desc))
            
            outfile_path_FR_mask= os.path.join(data_obj.FR_masks_dir, "{}_{}.tiff".format(mcd_root,desc))
            outfile_path_FR_removed_mask= os.path.join(data_obj.FR_removed_dir, "{}_{}.tiff".format(mcd_root,desc))
            
            outfile_path_final_segmented_mask= os.path.join(data_obj.final_segmented_masks_dir, "{}_{}.tiff".format(mcd_root,desc))
            
            outfile_path_quality_metrics= os.path.join(data_obj.metrics_dir, "{}_{}.csv".format(mcd_root,desc))
            outfile_path_fiber_metrics= os.path.join(data_obj.metrics_dir, "{}_{}_perFiber.csv".format(mcd_root,desc))
            
            segmentor_obj = segmentor.segmentor(membrane_img, mass_img,manual_mask,outfile_path_red_green,outfile_path_raw_fiber_segmentation,outfile_path_raw_segmented_masks_stardist,outfile_path_raw_segmented_masks_cellpose,outfile_path_NTM_mask,outfile_path_FAM_mask,outfile_path_FR_mask,outfile_path_final_segmented_mask)
            
            red_green_image = segmentor_obj.red_green_image_maker()   
            cv2.imwrite(outfile_path_red_green,red_green_image) 
            
            fiber_mask,fiber_mask_stardist, fiber_mask_cellpose  = segmentor_obj.fiber_segmentator()
            cv2.imwrite(outfile_path_raw_fiber_segmentation,fiber_mask)
            cv2.imwrite(outfile_path_raw_segmented_masks_stardist,fiber_mask_stardist)
            cv2.imwrite(outfile_path_raw_segmented_masks_cellpose,fiber_mask_cellpose)
            
              
            NTM_mask, fiber_metrics = segmentor_obj.NTM_mask_maker()
            cv2.imwrite(outfile_path_NTM_mask ,NTM_mask)
            np.savetxt(outfile_path_fiber_metrics,fiber_metrics,delimiter=",")
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
            cv2.imwrite(outfile_path_final_segmented_mask ,final_segmented_masks)
            
            if manual_mask is not None:
                quality_metrics_array= segmentor_obj.quality_metrics_calculator()
                np.savetxt(outfile_path_quality_metrics,quality_metrics_array, delimiter=",")
            
print('All segmentations complete, please check the respective folders')
    
