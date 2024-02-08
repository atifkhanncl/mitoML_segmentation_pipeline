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

datatype = "IMC"
#secondary_masks = ["NTM","FAM","FR"]
secondary_masks = []

if datatype == "QIF":
    is_qif= True
    is_imc= False
    cv_mode= 0
elif datatype == "IMC":
    is_imc= True
    is_qif= False
    cv_mode= -1
else:
    print("Need a valid datatype...  Exiting.")
    raise SystemExit(0)
    
membrane_list, mass_list = data_obj.get_image_list()
for infile in tqdm(membrane_list, desc = "Input Images Processed"):
    start = time.time()
    infile_mass_channel_path= os.path.join(data_obj.mass_channel_dir, infile)
    infile_membrane_channel_path= os.path.join(data_obj.membrane_channel_dir, infile)
    infile_manual_sample_mask_path= os.path.join(data_obj.manual_sample_masks_dir, infile.split('.')[0] + '.tiff')
    
    outfile_path_red_green= os.path.join(data_obj.red_green_images_dir, infile.split('.')[0] + '.tiff')
    
    outfile_path_raw_fiber_segmentation= os.path.join(data_obj.raw_segmented_masks_dir, infile.split('.')[0] + '.tiff')        
    
    mass_img= cv2.imread(infile_mass_channel_path,cv_mode)
    membrane_img= cv2.imread(infile_membrane_channel_path,cv_mode)
    manual_mask= np.zeros(mass_img.shape[0:2]).astype(np.uint16)

    segmentor_obj = segmentor.segmentor(membrane_img, mass_img,manual_mask,outfile_path_red_green,outfile_path_raw_fiber_segmentation,"","","","",model_match_threshold,is_imc,is_qif)
    
    red_green_image = segmentor_obj.red_green_image_maker()   
    cv2.imwrite(outfile_path_red_green,red_green_image) 
    print('\n Red-Green contrasted '+infile.split('.')[0]+' image completed  ')
    start_raw = time.time()
    fiber_mask = segmentor_obj.fiber_segmentator()
    cv2.imwrite(outfile_path_raw_fiber_segmentation,fiber_mask)
    stop_raw = time.time()
    t_raw= stop_raw-start_raw
    print('Raw segmented masks for '+infile.split('.')[0]+' image completed in  '+ str(t_raw)  + ' seconds')
    
print('All segmentations complete, please check the respective folders')
    
