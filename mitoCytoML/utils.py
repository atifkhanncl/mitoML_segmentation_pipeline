from PIL import Image
import PIL.ImageOps  
import numpy as np
import os
import cv2
import csv
Image.MAX_IMAGE_PIXELS = None
from skimage import exposure, img_as_float,img_as_uint
from scipy.spatial.distance import cdist
from csbdeep.utils import normalize



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
    img_BGR= cv2.imread(img_path,-1)
    #img_RGB=cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    return img_BGR

def contrast_stretching_quantiles(channel,plow,phigh):
    channel= img_as_float(channel)
    plower, phigher = np.percentile(channel, (plow, phigh))
    img_contrast_stretched = img_as_uint(exposure.rescale_intensity(channel, in_range=(plower, phigher)))
    return(img_contrast_stretched)

def contrast_stretching_adap(channel, c_limit):
    channel= img_as_float(channel)
    contrast_streched_channel = img_as_uint(exposure.equalize_adapthist(channel, clip_limit=c_limit))
    return(contrast_streched_channel)

def image_normalizer(img):
    axis_norm = (0,1)
    img_norm = normalize(img, 1.0,99.8, axis=axis_norm)
    return img_norm

def make_merge_channels(r,g):
    b = np.zeros(r.shape)
    rgb_uint8 = (np.dstack((b,g,r))*255.999 ) .astype(np.uint8) 
    return rgb_uint8

def makepseudo(arr,minpercent=5,maxpercent=95):
    minval,maxval = np.percentile(arr.flatten(),[minpercent,maxpercent])
    if(maxval>minval):
        arr = np.maximum(np.minimum(arr,maxval),minval)
        arr = arr-minval
        arrf = np.array(arr,dtype=np.float32)
        arrf = np.array(np.round(255.0*np.minimum(1.0,arrf/(maxval-minval))),dtype=np.uint8)
    else:
        arrf = np.array(arr[:,:],dtype=np.uint8)
    return(arrf)

def make_merge_channels_16bit(r,g):
    b = np.zeros(r.shape)
    bgr_uint16 = (np.dstack((b,g,r)) ) .astype(np.uint16) 
    return bgr_uint16

def red_green_image_maker(r,g ):
    red_green_image= make_merge_channels_16bit(makepseudo(r),makepseudo(g))
    return red_green_image

def red_green_image_maker_without_contrast(r,g):
    red_green_img_no_contrasting = make_merge_channels_16bit(r,g)
    return red_green_img_no_contrasting

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
