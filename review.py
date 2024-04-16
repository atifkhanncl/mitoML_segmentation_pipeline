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
key = input("\nIs this an IMC section (y/n, then press enter)?")
if key =="y":
    isIMC = True
elif key =="n":
    isIMC = False
else:
    print("Section type response does not make sense...")
    
fname = fname+".tiff"

# Set areamin & areamax to -1 to disable filtering objects by area
areamin, areamax = 210, 27000 # Taken from Atif's NTM filter
IF20x = 0.325*0.325 #um2 per pixel in IF sections imaged at 20x magnification?

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
    
    viewer.open(os.path.join(red_green_images_dir,fname), layer_type="image", multiscale=False, rgb=True,name="red_green", visible=False)   
    if len(all_fiber_ids_to_remove)!=0:
        layer_final= viewer.add_labels(final_mask, multiscale=False,name="final", visible=False)
    if os.path.isfile(os.path.join(FAM_masks_dir,fname)):
        layer_FAM= viewer.add_labels(FAM_mask, multiscale=False,name="FAM",visible=False)
    if os.path.isfile(os.path.join(NTM_masks_dir,fname)):
        layer_NTM= viewer.add_labels(NTM_mask, multiscale=False,name="NTM",visible=False)
    if os.path.isfile(os.path.join(FR_masks_dir,fname)):
        layer_FR= viewer.add_labels(FR_mask, multiscale=False,name="Folded_Regions",visible=False),

    dup_RG=(viewer.layers['red_green']).data
    membrane_chan=dup_RG[:, :, 0]
    cytoplasm_chan=dup_RG[:, :, 1]
    viewer.add_image(cytoplasm_chan, multiscale=False, rgb=False, colormap="green")
    viewer.add_image(membrane_chan, multiscale=False, rgb=False, blending = "additive", colormap="red")
    
    layer_raw_segmentation=viewer.add_labels(raw_mask, multiscale=False,name="Segmented_Fibres",
                                              blending = "translucent",opacity=0.7,visible=True)

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

if isIMC:
    # Assume 1px = 1um2
    load_all_layers(viewer,fname,areamin=areamin,areamax=areamax)
else:
    # Convert pixels to um2
    load_all_layers(viewer,fname,areamin=areamin/IF20x, areamax=areamax/IF20x)

napari.run()

