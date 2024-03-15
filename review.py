import napari
from mitoCytoML.utils import *
from mitoCytoML.data import *
import glob
import os
import numpy as np
import pathlib

np.random.seed(1)

# Get all possible input filenames:
allfs = list(pathlib.Path(os.path.join(raw_segmented_masks_dir)).glob('*.tiff'))
allstems = [p.stem for p in allfs]
for i,s in enumerate(allstems):
    print(str(i),s)
fname= input("\nType the name of the section to review (or its index number) and press enter: ")
if fname.isdigit() and int(fname)<len(allstems):
    fname = allstems[int(fname)]
else:
    print("Bad section name, try again...")
fname= fname+".tiff"


viewer = napari.Viewer()


def load_all_layers(viewer,fname,areamin=-1,areamax=-1):

    if os.path.isfile(os.path.join(NTM_masks_dir,fname)):
        NTM_fiber_ids= get_unique_labels_in_mask(read_image_io(os.path.join(NTM_masks_dir,fname)))
    else:
        NTM_fiber_ids=[]
    if os.path.isfile(os.path.join(FAM_masks_dir,fname)):
        FAM_fiber_ids= get_unique_labels_in_mask(read_image_io(os.path.join(FAM_masks_dir,fname)))
    else:
        FAM_fiber_ids=[]
    if os.path.isfile(os.path.join(FR_masks_dir,fname)):
        FR_fiber_ids= get_unique_labels_of_ovelapping_pixels(read_image_io(os.path.join(raw_segmented_masks_dir,fname)), read_image_io(os.path.join(FR_masks_dir,fname)))
    else:
        FR_fiber_ids=[]
    all_fiber_ids_to_remove= all_fiber_ids_to_remove= np.union1d(np.union1d(NTM_fiber_ids , FAM_fiber_ids ), FR_fiber_ids) # Could have just used python sets here? 
    
    raw_mask= io.imread(os.path.join(raw_segmented_masks_dir,fname))
    if areamin>=0 and areamax > areamin:
        bc = np.bincount(raw_mask.flatten())
        fibcols = [i for i,count in enumerate(bc) if (count>areamin and count<areamax)]
        remove = np.logical_not(np.isin(raw_mask,fibcols))
        raw_mask[remove]=0
    else:
        print("Not filtering any objects from raw mask")        

    if os.path.isfile(os.path.join(FR_masks_dir,fname)):
        FR_mask= read_image_io(os.path.join(FR_masks_dir,fname))
    NTM_mask= mask_selected_fibers_array_retain (read_image_io(os.path.join(raw_segmented_masks_dir,fname)), NTM_fiber_ids)
    FAM_mask= mask_selected_fibers_array_retain (read_image_io(os.path.join(raw_segmented_masks_dir,fname)), FAM_fiber_ids)
    final_mask= mask_selected_fibers_array_remove (read_image_io(os.path.join(raw_segmented_masks_dir,fname)), all_fiber_ids_to_remove)
    
    viewer.open(os.path.join(red_green_images_dir,fname), layer_type='image', multiscale=False, rgb=True,name="red_green")   
    if len(all_fiber_ids_to_remove)!=0:
        layer_final= viewer.add_labels(final_mask, multiscale=False,name="final", visible=False)
    if os.path.isfile(os.path.join(FAM_masks_dir,fname)):
        layer_FAM= viewer.add_labels(FAM_mask, multiscale=False,name="FAM",visible=False)
    if os.path.isfile(os.path.join(NTM_masks_dir,fname)):
        layer_NTM= viewer.add_labels(NTM_mask, multiscale=False,name="NTM",visible=False)
    if os.path.isfile(os.path.join(FR_masks_dir,fname)):
        layer_FR= viewer.add_labels(FR_mask, multiscale=False,name="Folded_Regions",visible=False)
    layer_raw_segmentation= viewer.add_labels(raw_mask, multiscale=False,name="Segmented_Fibres",visible=False)

    if os.path.isfile(os.path.join(NTM_masks_dir,fname)):
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

    if os.path.isfile(os.path.join(FAM_masks_dir,fname)):
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

    @layer_raw_segmentation.mouse_double_click_callbacks.append
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

    msg = 'All labels removed but this action is not final until saved. To save press s'
    viewer.status = msg
    print(msg)

@viewer.bind_key('s')
def save_layers(viewer):
    viewer.layers['Segmented_Fibres'].save(os.path.join(raw_segmented_masks_dir,fname))
    if 'NTM' in viewer.layers: viewer.layers['NTM'].save(os.path.join(NTM_masks_dir,fname))
    if 'FAM' in viewer.layers: viewer.layers['FAM'].save(os.path.join(FAM_masks_dir,fname))
    if 'Folded_Regions' in viewer.layers: viewer.layers['Folded_Regions'].save(os.path.join(FR_masks_dir,fname))
    if 'final' in viewer.layers: viewer.layers['final'].save(os.path.join(final_segmented_masks_dir,fname))

    viewer.layers.clear() 

    load_all_layers(viewer,fname)


    msg = 'All changes saved'
    viewer.status = msg
    print(msg)

@viewer.bind_key('c')
def clear_layers(viewer):
    viewer.layers.clear() 
    
print("Reviewing ",fname)
# Set areamin & areamax to -1 to disable filtering objects by area
load_all_layers(viewer,fname,areamin=210,areamax=27000)  

napari.run()

