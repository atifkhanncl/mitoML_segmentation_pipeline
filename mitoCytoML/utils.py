from PIL import Image
import PIL.ImageOps  
import numpy as np
import os
import cv2
import csv
Image.MAX_IMAGE_PIXELS = None
from skimage import exposure, img_as_float
from scipy.spatial.distance import cdist



def images_list(directory):
    img_list = [f for f in
                os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))]
    return img_list

def check_file_exist(file_path):
    return os.path.isfile(file_path)

def dir_create(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(img_path):
    img= cv2.imread(img_path,-1)
    return img
def contrast_stretching_quantiles(arr,plow,phigh):
    arr =1.0 * arr #float
    plower, phigher = np.percentile(arr, (plow, phigh))
    img_contrast_stretched = exposure.rescale_intensity(arr, in_range=(plower, phigher))
    return(img_contrast_stretched)
def contrast_stretching_adap(channel, c_limit=0.0001):
    channel= img_as_float(channel)
    contrast_streched_channel = exposure.equalize_adapthist(channel, clip_limit=c_limit)
    return(contrast_streched_channel)
def make_merge_channels(r,g):
    b = np.zeros(r.shape)
    rgb_uint8 = (np.dstack((b,g,r))*255.999 ) .astype(np.uint8) 
    return rgb_uint8
def red_green_image_maker(r,g, c_limit_membrane=0.1, c_limit_mass=0.1 ):
    red_green_image= make_merge_channels(contrast_stretching_adap(r, c_limit= c_limit_membrane),contrast_stretching_adap(g, c_limit= c_limit_mass ))
    return red_green_image

def pixels_remover(pixels_to_remove,from_img):
    #pixels_removed_img = from_img[pixels_to_remove==255] = 0
    from_img[pixels_to_remove == 255] = 0  
    
    return from_img

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def ind_fiber_image_maker(rawim, maskim):
    maskarr = np.array(maskim,dtype=np.uint16)
    rawarr = np.array(rawim,dtype=np.uint8)
    unique_colours = np.unique(maskarr)
    end_row = maskarr.shape[0]
    end_column = maskarr.shape[1]
    # Discard black background
    unique_colours = [c for c in unique_colours if c!=0]
    ind_fibers_list = []
    for colour in unique_colours:
        # Find pixels belonging to specific fibre
        fibmask = maskarr==colour
        bbox = bbox2(fibmask)
        start_row= (bbox[0] -3) if bbox[0]>2 else bbox[0]
        stop_row= (bbox[1] +3) if bbox[1]<(end_row -2) else bbox[1]
        start_column= (bbox[2] -3) if bbox[2]>2 else bbox[2]
        stop_column= (bbox[3] +3) if bbox[3]<(end_column -2) else bbox[3]
        # Crop arrays to bounding box
        #fibarr = rawarr[bbox[0]:bbox[1],bbox[2]:bbox[3],:]
        #fibmask = fibmask[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        fibarr = np.copy(rawarr[start_row:stop_row,start_column:stop_column,:])
        # Convert cropped array to image and save to disk
        if(fibarr.size>0):
            fibmask2 = fibmask[start_row:stop_row,start_column:stop_column]
            # change mask pixel to 255
            fibmask2[fibmask2 ==colour] = 255
            # Delete any non-fibre pixels that remain in bounding box
            fibarr[np.logical_not(fibmask2)] = [0,0,0]
            ind_fibers_list.append(fibarr)
            #fibim = Image.fromarray(fibarr)
            #img_path = os.path.join(out_dir,filename+'_'+str(colour)+ '.tiff')
            #fibim.save(img_path)
    return ind_fibers_list, unique_colours
