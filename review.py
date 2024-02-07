import napari
from mitoCytoML.utils import *
from mitoCytoML.data import *
import glob
import os
import numpy as np
np.random.seed(1)

fname= input("Type the filename to review and enter: ")
fname= fname+".tiff"


viewer = napari.Viewer()


def load_all_layers(viewer,fname):
    
    NTM_fiber_ids= get_unique_labels_in_mask(read_image_io(os.path.join(NTM_masks_dir,fname)))
    FAM_fiber_ids= get_unique_labels_in_mask(read_image_io(os.path.join(FAM_masks_dir,fname)))
    FR_fiber_ids= get_unique_labels_of_ovelapping_pixels(read_image_io(os.path.join(raw_segmented_masks_dir,fname)), read_image_io(os.path.join(FR_masks_dir,fname)))
    all_fiber_ids_to_remove= all_fiber_ids_to_remove= np.union1d(np.union1d(NTM_fiber_ids , FAM_fiber_ids ), FR_fiber_ids) 
    
    raw_mask= read_image_io(os.path.join(raw_segmented_masks_dir,fname))
    FR_mask= read_image_io(os.path.join(FR_masks_dir,fname))
    NTM_mask= mask_selected_fibers_array_retain (read_image_io(os.path.join(raw_segmented_masks_dir,fname)), NTM_fiber_ids)
    FAM_mask= mask_selected_fibers_array_retain (read_image_io(os.path.join(raw_segmented_masks_dir,fname)), FAM_fiber_ids)
    final_mask= mask_selected_fibers_array_remove (read_image_io(os.path.join(raw_segmented_masks_dir,fname)), all_fiber_ids_to_remove)
    
    viewer.open(os.path.join(red_green_images_dir,fname), layer_type='image', multiscale=False, rgb=True,name="red_green")   
    layer_final= viewer.add_labels(final_mask, multiscale=False,name="final", visible=False)
    layer_FAM= viewer.add_labels(FAM_mask, multiscale=False,name="FAM",visible=False)
    layer_NTM= viewer.add_labels(NTM_mask, multiscale=False,name="NTM",visible=False)
    layer_FR= viewer.add_labels(FR_mask, multiscale=False,name="Folded_Regions",visible=False)
    layer_raw_segmentation= viewer.add_labels(raw_mask, multiscale=False,name="Segmented_Fibres",visible=False)
       
    @layer_NTM.mouse_double_click_callbacks.append
    def update_required_layers(layer, event):
        data_coordinates = layer.world_to_data(event.position)
        coords = np.round(data_coordinates).astype(int)
        val = layer.get_value(data_coordinates)
        if val is None:
            return
        if val != 0:
            data = layer.data
            data[data==val] = 0
            layer.data = data

        else:
            data_raw = layer_raw_segmentation.data
            val_raw= data_raw[coords[0],coords[1]]
            if val_raw is None:
                return
            if val_raw !=0:
                data = layer.data
                data[data_raw==val_raw] = val_raw
                layer.data = data

            else:
                print(f'clicked at {coords} on background which is ignored')   

    @layer_FAM.mouse_double_click_callbacks.append
    def update_required_layers(layer, event):
        data_coordinates = layer.world_to_data(event.position)
        coords = np.round(data_coordinates).astype(int)
        val = layer.get_value(data_coordinates)
        if val is None:
            return
        if val != 0:
            data = layer.data
            data[data==val] = 0
            layer.data = data

        else:
            data_raw = layer_raw_segmentation.data
            val_raw= data_raw[coords[0],coords[1]]
            if val_raw is None:
                return
            if val_raw !=0:
                data = layer.data
                data[data_raw==val_raw] = val_raw
                layer.data = data

            else:
                print(f'clicked at {coords} on background which is ignored') 

                
@viewer.bind_key('r')
def reset_layer(viewer):
    layer_data= list(viewer.layers.selection)[0].data
    layer_data[layer_data !=0] = 0
    (viewer.layers.selection)[0] = layer_data

    msg = 'All labels removed but this action is not final util saved. To save press s'
    viewer.status = msg
    print(msg)

@viewer.bind_key('s')
def save_all_layers(viewer):
    viewer.layers['Segmented_Fibres'].save(os.path.join(raw_segmented_masks_dir,fname))
    viewer.layers['NTM'].save(os.path.join(NTM_masks_dir,fname))
    viewer.layers['FAM'].save(os.path.join(FAM_masks_dir,fname))
    viewer.layers['Folded_Regions'].save(os.path.join(FR_masks_dir,fname))
    viewer.layers['final'].save(os.path.join(final_segmented_masks_dir,fname))


    viewer.layers.clear() 

    load_all_layers(viewer,fname)


    msg = 'All changes saved'
    viewer.status = msg
    print(msg)

@viewer.bind_key('c')
def clear_layers(viewer):
    viewer.layers.clear() 

impaths = glob.glob(os.path.join(red_green_images_dir, "*.tiff"))
fnames = [os.path.basename(impath) for impath in impaths]    
if fname not in fnames:
    print ("Invalid filename")
    
else:
    load_all_layers(viewer,fname)    

napari.run()

