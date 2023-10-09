import napari
import glob
import os

output_folder = "C:\\Users\\Conor\\Documents\\parseMCD\\output"

im = "red_green_images"

impaths = glob.glob(os.path.join(output_folder,im,"*.tiff"))
fnames = [os.path.basename(impath) for impath in impaths]

for fname in fnames:
    viewer = napari.Viewer()
    viewer.open(os.path.join(output_folder,im,fname), layer_type='image', multiscale=False, rgb=True,name="red_green")
    viewer.open(os.path.join(output_folder,"Step_1","raw_segmented_masks","cellpose",fname), layer_type='labels', multiscale=False,name="cellpose",visible=False)
    viewer.open(os.path.join(output_folder,"Step_1","raw_segmented_masks","stardist",fname), layer_type='labels', multiscale=False,name="stardist",visible=False)
    viewer.open(os.path.join(output_folder,"Step_2A","NTM_masks",fname), layer_type='labels', multiscale=False,name="NTM",visible=False)
    viewer.open(os.path.join(output_folder,"Step_2A","NTM_removed_masks",fname), layer_type='labels', multiscale=False,name="NTM_removed",visible=False)
    viewer.open(os.path.join(output_folder,"Step_2B","FAM_masks",fname), layer_type='labels', multiscale=False,name="FAM",visible=False)
    viewer.open(os.path.join(output_folder,"Step_2B","FAM_removed_masks",fname), layer_type='labels', multiscale=False,name="FAM_removed",visible=False)
    viewer.open(os.path.join(output_folder,"Step_3","FR_masks",fname), layer_type='labels', multiscale=False,name="FR",visible=False)
    viewer.open(os.path.join(output_folder,"Step_3","FR_removed_fiber_mask",fname), layer_type='labels', multiscale=False,name="FR_removed",visible=False)
    viewer.open(os.path.join(output_folder,"Step_4_Final_segmented_masks",fname), layer_type='labels', multiscale=False,name="final")
