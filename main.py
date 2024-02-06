from mitoCytoML import segmentor
from mitoCytoML import data
import os
import cv2
import numpy as np
import time
from tqdm import tqdm
import torch
data_obj = data.data_obj ()
data_obj.create_dir_structure()
model_match_threshold=27

if torch.cuda. is_available():
    print ("GPU and CUDA detected and will be used")
else:
    print ("GPU and CUDA not detected and CPU will be used")
    
response_1= input("Types of images: For QIF press'Q' and enter or For IMC press 'I' and enter: ")
if response_1 =='Q' or 'q':
    is_qif= True
    is_imc= False
    cv_mode= 0
if response_1 == 'I' or 'i':
    is_imc= True
    is_qif= False
    cv_mode= -1
if response_1 not in [ 'Q', 'q' ,'I' ,'i']:
    print('incorrect option selected '+response_1+' , start again')


    
response_3 = input("Please place required raw images into the respective folder then press 'y' and enter ")
if response_3 == 'y' or 'Y':
    membrane_list, mass_list = data_obj.get_image_list()
    for infile in tqdm(membrane_list, desc = "Input Images Processed"):
        start = time.time()
        infile_mass_channel_path= os.path.join(data_obj.mass_channel_dir, infile)
        infile_membrane_channel_path= os.path.join(data_obj.membrane_channel_dir, infile)
        infile_manual_sample_mask_path= os.path.join(data_obj.manual_sample_masks_dir, infile.split('.')[0] + '.tiff')
        
        outfile_path_red_green= os.path.join(data_obj.red_green_images_dir, infile)
        
        outfile_path_raw_fiber_segmentation= os.path.join(data_obj.raw_segmented_masks_dir, infile.split('.')[0] + '.tiff')        
        #outfile_path_raw_segmented_masks_stardist= os.path.join(data_obj.raw_segmented_masks_stardist_dir, infile.split('.')[0] + '.tiff')
        #outfile_path_raw_segmented_masks_cellpose= os.path.join(data_obj.raw_segmented_masks_cellpose_dir, infile.split('.')[0] + '.tiff') 
        
        outfile_path_NTM_mask= os.path.join(data_obj.NTM_masks_dir, infile.split('.')[0] + '.tiff')
        outfile_path_NTM_removed_mask= os.path.join(data_obj.NTM_removed_masks_dir, infile.split('.')[0] + '.tiff')
        
        outfile_path_FAM_mask= os.path.join(data_obj.FAM_masks_dir, infile.split('.')[0] + '.tiff')
        outfile_path_FAM_removed_mask= os.path.join(data_obj.FAM_removed_masks_dir, infile.split('.')[0] + '.tiff')
        
        outfile_path_FR_mask= os.path.join(data_obj.FR_masks_dir, infile.split('.')[0] + '.tiff')
        outfile_path_FR_removed_mask= os.path.join(data_obj.FR_removed_dir, infile.split('.')[0] + '.tiff')
        
        outfile_path_final_segmented_mask= os.path.join(data_obj.final_segmented_masks_dir, infile.split('.')[0] + '.tiff')
        
        outfile_path_quality_metrics= os.path.join(data_obj.metrics_dir, infile.split('.')[0]+'.csv')
        outfile_path_fiber_metrics= os.path.join(data_obj.metrics_dir, infile.split('.')[0]+'_perFiber.csv')
        
        mass_img= cv2.imread(infile_mass_channel_path,cv_mode)
        membrane_img= cv2.imread(infile_membrane_channel_path,cv_mode)
        manual_mask= cv2.imread(infile_manual_sample_mask_path,-1)
        if manual_mask is None:
            manual_mask= np.zeros(mass_img.shape[0:2]).astype(np.uint16)
        
        segmentor_obj = segmentor.segmentor(membrane_img, mass_img,manual_mask,outfile_path_red_green,outfile_path_raw_fiber_segmentation,outfile_path_NTM_mask,outfile_path_FAM_mask,outfile_path_FR_mask,outfile_path_final_segmented_mask,model_match_threshold,is_imc,is_qif)
        
        red_green_image = segmentor_obj.red_green_image_maker()   
        cv2.imwrite(outfile_path_red_green,red_green_image) 
        print('\n Red-Green contrasted '+infile.split('.')[0]+' image completed  ')
        start_raw = time.time()
        fiber_mask = segmentor_obj.fiber_segmentator()
        cv2.imwrite(outfile_path_raw_fiber_segmentation,fiber_mask)
        #cv2.imwrite(outfile_path_raw_segmented_masks_stardist,fiber_mask_stardist)
        #cv2.imwrite(outfile_path_raw_segmented_masks_cellpose,fiber_mask_cellpose)
        stop_raw = time.time()
        t_raw= stop_raw-start_raw
        print('Raw segmented masks for '+infile.split('.')[0]+' image completed in  '+ str(t_raw)  + ' seconds')
        
        start_ntm=time.time()  
        NTM_mask, fiber_metrics = segmentor_obj.NTM_mask_maker()
        cv2.imwrite(outfile_path_NTM_mask ,NTM_mask)
        np.savetxt(outfile_path_fiber_metrics,fiber_metrics,delimiter=",")
        NTM_removed_mask = segmentor_obj.NTM_remover()
        cv2.imwrite(outfile_path_NTM_removed_mask ,NTM_removed_mask)
        stop_ntm= time.time()
        t_ntm= stop_ntm-start_ntm
        print('NTM mask for '+infile.split('.')[0]+' image completed  in  '+ str(t_ntm) + ' seconds')
        
        start_fam=time.time()
        FAM_mask = segmentor_obj.FAM_mask_maker()
        cv2.imwrite(outfile_path_FAM_mask ,FAM_mask)
        FAM_removed_mask = segmentor_obj.FAM_remover()
        cv2.imwrite(outfile_path_FAM_removed_mask ,FAM_removed_mask)
        stop_fam= time.time()
        t_fam=stop_fam-start_fam
        print('FAM mask for '+infile.split('.')[0]+' image completed in  '+  str(t_fam) + ' seconds')
        
        start_fr= time.time()
        FR_mask = segmentor_obj.folded_region_segmentor()
        cv2.imwrite(outfile_path_FR_mask ,FR_mask)
        stop_fr=time.time()
        t_fr= stop_fr-start_fr
        print('FR mask for '+infile.split('.')[0]+' image completed in  '+ str(t_fr) + ' seconds')
        
        FR_removed_mask = segmentor_obj.folded_region_remover()
        cv2.imwrite(outfile_path_FR_removed_mask ,FR_removed_mask)
      
        start_final=time.time()
        final_segmented_masks = segmentor_obj.non_analysable_fibers_remover()
        cv2.imwrite(outfile_path_final_segmented_mask ,final_segmented_masks)
        stop_final= time.time()
        t_final=stop_final-start_final
        print('Final mask for '+infile.split('.')[0]+' image completed  in  ' + str(t_final)+ ' seconds')
        
        if manual_mask is not None:
            start_q= time.time()
            quality_metrics_array= segmentor_obj.quality_metrics_calculator()
            np.savetxt(outfile_path_quality_metrics,quality_metrics_array, delimiter=",")
            stop_q= time.time()
            t_q=stop_q-start_q
            
            print('computing quality metrics for '+infile.split('.')[0]+' image completed  in  ' + str(t_q) + ' seconds')
        stop = time.time()
        t= stop-start
        print( 'processing of '+infile.split('.')[0]+ 'completed in '+ str(t) + ' seconds')
            
            
        
print('All segmentations complete, please check the respective folders')
    