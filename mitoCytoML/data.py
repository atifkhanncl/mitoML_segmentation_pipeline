from PIL import Image
import PIL.ImageOps  
import numpy as np
import os
import cv2
import csv
from mitoCytoML import utils

raw_images_dir = "./raw_images/"
mass_channel_dir= raw_images_dir+'/mass_channel_images/'
membrane_channel_dir= raw_images_dir+'/membrane_channel_images/'
FR_removed_dir= "./FR_removed_RG_images/"
raw_segmented_masks_dir ="./raw_segmented_masks/"
individual_fibers_dir ='./individual_fibers/'
final_segmented_masks_dir= "./final_segmented_masks/"
model_FR_dir = "./model_FR/"
FR_model_name='FR_model'
model_segmentation_dir= "./model_fiber_segmentation/"
model_classification_dir= "./model_fiber_classification/"
c_limit_mass=0.01
c_limit_membrane= 0.01