# mitoML_segmentation_pipeline
Machine learning pipeline to segment muscle fibers from IMC and IF images 



# Using mitocytoML

## Getting Started

### Prerequisites

Python 3.8 - 3.10  

*We highly recommand that uses the version between 3.8 and 3.10. With these versions, dependencies can be installed properly.


### Installation

After cloning the repo into a new directory, make a virtual environment called 'venv', activate it, and install the dependencies using pip tool, e.g.

``` shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Note: the requirements file contains all the dependencies needed, and it works well when the version of python is in the mentioned range.**

Once finished   

```
deactivate
```

## Segmenting fibers

mitocytoML can be utilised to segment muscle fibers from IMC and IF image data either just running 'mitoCytoML.py' or with 3 simple steps:

1) Place raw images representing mass and membrane of the fibers in respective folders within 'raw_images' folder. 
2) Load the mitocytoML `segmentor` and 'data' objects
3) Invoke `data.data_obj()` to load data object and 'segmentor.segmentor()' to access all segmentor functions and attributes


We handle the rest.

### Synthetic Image Evaluator Example

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
