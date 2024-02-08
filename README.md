# mitoML_segmentation_pipeline
Machine learning pipeline to segment muscle fibers from IMC and IF images 



# Using mitocytoML

## Getting Started

### Prerequisites

Python 3.8 or above

*We highly recommand that uses the version > 3.9 <3.10 . With these versions, dependencies can be installed properly.


### Installation

After cloning the repo into a new directory, install dependencies

``` shell
pip install -r requirements.txt
```

**Note: the requirements file contains all the dependencies needed, and it works well when the version of python is in the mentioned range.**

## Using the mitocytoML
Step 1: using terminal navigate to repo and and type 'python main.py'

Step 2: it will prompt 'Please place required raw images into the respective folder then press 'y' and enter'. Please place raw tiff images of memebrane and mass channels in respective folders within input. Also, if you want to know the quality of annotations then please manually annotate few myofibres and place the mask in 'manual_sample_masks' within input folder.

Step 3: follow the prompts on the terminal.

Step 4: At completion. Find all masks in output folder. If you're not happy with any mask, please see below section.

## Reviewing mitocytoML outputs
You can review and correct all generated masks using review.py


Step 1: using terminal navigate to repo and and type 'python review.py'

Step 2: Type filename and enter. This will open a napari session with 6 layers with image on ‘view’ mode and all masks on hide mode

Step 3: Review ‘Segmented_Fibres’ layer
-  1.  You can change borders of existing fibres by selecting fiber id and using brush or eraser tool
-  2.  To add a new fibre press ‘m’ key and annotate new fibre using brush and bucket tool
-  3.  To save corrections press ‘s’ key (please mind 'save' is expensive operation can take upto 30 seconds)

Step 4: Review ‘Folded_Regions’ layer
-  1.  To correct folded region mask, use eraser, brush tools
-  2.  If you want to reject all existing annotations press ‘r’ key
-  3.  To save corrections press ‘s’ key (please mind 'save' is expensive operation can take upto 30 seconds)

Step 5: Review ‘NTMs’ layer
-  1.  To change the classification of any fibres, double click on the fibre
-  2.  If you want to reject all existing annotations press ‘r’ key
-  3.  To save corrections press ‘s’ key (please mind 'save' is expensive operation can take upto 30 seconds)
   

Step 6: Review ‘FAMs’ layer
-  1.  To change the classification of any fibres, double click on the fibre
-  2.  If you want to reject all existing annotations press ‘r’ key
-  3.  To save corrections press ‘s’ key (please mind 'save' is expensive operation can take upto 30 seconds)

Step 7: Review ‘final’ layer
-  Upon reviewing step 1-6, if you're happy with all layers please press ‘s’ key. This will save all corrections and compute new ‘final’ mask.
       

## Segmenting fibers

mitocytoML can be utilised to segment muscle fibers from IMC and IF image data either by just running 'main.py' or with 3 simple steps:

1) Place raw images representing mass and membrane of the fibers in respective folders within 'raw_images' folder. 
2) Load the mitocytoML `segmentor` and 'data' objects
3) Invoke `data.data_obj()` to load data object and 'segmentor.segmentor()' to access all segmentor functions and attributes


We handle the rest.

### GPU

In order to use a GPU:

1. Uninstall the PyTorch version that gets installed by default with Cellpose:

        pip uninstall torch

2. Make sure your have up-to-date drivers for your NVIDIA card installed.

3. Re-install a GPU version of PyTorch via conda using a command that you can find [here](https://pytorch.org/get-started/locally/) (this takes care of the cuda toolkit, cudnn etc. so **no need to install manually anything more than the driver**). The command will look like this (but you should generate your own version of the command below using the tool on that webpage):

        pip3.9 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
## Viewing the resulted images and masks

we encourage you use [napari] (https://napari.org/stable/) to view results.

### mitoCytoML Fiber Segmentor Example

Load in the segmentor object
```python
from mitoCytoML import segmentor
```

Load in your data as two images `size = [H,W,C]`:
```python
from mitoCytoML import data
data.data_obj ()
```

Intialise the fiber segmentor and feed in your data:

```python
img_segmentor = segmentor.segmentor(membrane_img, mass_img, outfile_path_red_green, outfile_path_folded_region)
```

Call your desired fuctions(s):
```
img_segmentor.red_green_image_maker()
img_segmentor.fiber_segmentator()
img_segmentor.folded_region_segmentor()
img_segmentor.folded_region_remover()
img_segmentor.fiber_analysability_classifier()
img_segmentor.non_analysable_fibers_remover()


## Contributing:
mitocytoML is an open source codebase.

#### Linting
This library uses `flake8` for linting. Please identify formatting errors in your code **before** pushing by running the following 

```
flake8 path/to/file/to/test
```

## Acknowledgements
This work was funded by the EPSRC Centre for Doctoral Training in Cloud Computing for Big Data and The Wellcome Centre for Mitochondrial Research.
