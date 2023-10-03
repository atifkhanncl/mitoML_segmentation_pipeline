from PIL import Image
import PIL.ImageOps  
import numpy as np
import os
import cv2
import csv
Image.MAX_IMAGE_PIXELS = None
from skimage import exposure, img_as_float,img_as_uint
from skimage.morphology import convex_hull_image
from scipy.spatial.distance import cdist
from csbdeep.utils import normalize



def file_list(directory):
    file_list = [f for f in
                os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))]
    return file_list

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
def make_merge_channels_8bit(r,g):
    b = np.zeros(r.shape)
    bgr_uint8 = (np.dstack((b,g,r)) ) .astype(np.uint8) 
    return bgr_uint8

def red_green_image_maker(r,g ):
    red_green_image= make_merge_channels_8bit(makepseudo(r),makepseudo(g))
    return red_green_image

def red_green_image_maker_without_contrast(r,g):
    red_green_img_no_contrasting = make_merge_channels_16bit(r,g)
    return red_green_img_no_contrasting

def pixels_remover(pixels_to_remove,from_img):
    #pixels_removed_img = from_img[pixels_to_remove==255] = 0
    from_img[pixels_to_remove == 255] = 0  
    
    return from_img
def give_blank_mask(mask):    
    blank= np.zeros(mask.shape[0:2]).astype(np.uint16)
    return blank


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def fiber_at_border(mask):
    border_fiber= False
    if (any(pixel !=0 for pixel in mask[0,:-1]) or
        any(pixel !=0 for pixel in mask[:-1,-1]) or
        any(pixel !=0 for pixel in mask[-1,::-1]) or
        any(pixel !=0 for pixel in mask[-2:0:-1,0])):
        border_fiber= True
    return border_fiber 


def fiber_areas(mask,min_area=1):
    unique, counts = np.unique(mask[np.nonzero(mask)], return_counts=True)
    pixel_val_count= dict(zip(unique, counts))
    pixels_ids_and_area= {k:v for k, v in pixel_val_count.items() if v > min_area}
    return pixels_ids_and_area

def fiber_convexity(individual_fiber_mask, fiber_area):
    convexity = (np.count_nonzero(convex_hull_image(individual_fiber_mask)))/fiber_area
    
    return convexity

def add_border(mask, inflate_pixels=10):
    
    h,w = mask.shape
    Dw,Dh = inflate_pixels,inflate_pixels
    img = np.zeros((h+2*Dh,w+2*Dw))
    img[:,:] = mask.min()
    img[Dh:(Dh+h),Dw:(Dw+w)] = mask
    
    return img
def contour_finder (mask):
    
    # Find fibres from mask_GT
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,2)
    # Ensure only looking at holes inside contours...
    contours = [c for i,c in enumerate(contours) if hierarchy[0][i][3] != -1]
    return contours

def biggest_contour(mask, canny_thershold=100):
    
    mask_border= np.uint8(add_border(mask))
    mask_border_edge  = cv2.Canny(mask_border,canny_thershold, canny_thershold * 2)
    
    contours= contour_finder (mask_border_edge)
    contour_lengths=[]
    for contour in contours:
        contour_lengths.append(len(contour))
    biggest_cont= [] if len(contour_lengths)==0 else contours[contour_lengths.index(max(contour_lengths))]
    
    return biggest_cont

def aspect_ratio_for_contour (contour):
    
    ellipse = cv2.fitEllipse(contour)
    (xc,yc),(d_minor,d_major),angle = ellipse
    i= int(d_minor)
    j= int(d_major)
    aspect_ratio= calculate_aspect(i,j)
    
    
    return aspect_ratio, i, j

def fiber_aspect_ratio(individual_fiber_mask, canny_thershold=100):
    contour= biggest_contour(individual_fiber_mask, canny_thershold)
    aspect_ratio_e, width_e, length_e = (0,0,0) if len(contour) <10 else aspect_ratio_for_contour (contour)
    
    return aspect_ratio_e

def calculate_aspect(i: int, j: int):
    def gcd(a, b):
        """The GCD (greatest common divisor) is the highest number that evenly divides both width and height."""
        return a if b == 0 else gcd(b, a % b)
    length= max(i,j)
    width=min(i,j)

    r = gcd(width, length)
    x = int(length / r)
    y = int(width / r)
    aspect_ratio = x/y

    return aspect_ratio

def aspect_ratio_for_mask (mask):
    segmentation = np.nonzero(mask)
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))
            
        i= int(x_max-x_min)
        j= int(y_max-y_min)
        width= min(i,j)
        length= max(i,j)
        aspect_ratio= calculate_aspect(i,j)
        
    return aspect_ratio,width,length

def fiber_lengthSquaredbyArea_bbox(individual_fiber_mask, fiber_area):
    aspect_ratio_bbox, width_bbox, length_bbox = aspect_ratio_for_mask (individual_fiber_mask)
    lengthSquaredbyArea_bbox= (length_bbox *length_bbox)/fiber_area
    
    return lengthSquaredbyArea_bbox

def mask_selected_fibers_array_remove (mask_proper, fibres_to_remove_array):    
    
    blank= np.zeros(mask_proper.shape[0:2]).astype(np.uint16)
    mask_to_return= np.where(np.isin(mask_proper, fibres_to_remove_array), blank, mask_proper)
                      
    return mask_to_return

def mask_selected_fibers_array_retain (mask_proper, fibres_to_retain_array):
    
    blank= np.zeros(mask_proper.shape[0:2]).astype(np.uint16)
    mask_to_return= np.where(np.isin(mask_proper, fibres_to_retain_array), mask_proper, blank)
                       
    return mask_to_return

def from_mask_to_remove_mask (mask_proper, mask_remove):
    
    
    
    blank= np.zeros(mask_proper.shape[0:2]).astype(np.uint16)
    unique_fiber_ids_to_remove = np.unique(mask_proper[np.nonzero(mask_remove)])
    
    mask_to_return= np.where(np.isin(mask_proper, unique_fiber_ids_to_remove), blank, mask_proper)
    
                   
    return mask_to_return  
    
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
