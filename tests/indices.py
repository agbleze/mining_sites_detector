
#%%
from mining_sites_detector.data_preprocessor import get_tiff_img
import numpy as np
from glob import glob
import os
from mining_sites_detector.inputs import filepath, folder

#%%


#%%
img = get_tiff_img(path=filepath, 
                   return_all_bands=True, 
                   normalize_bands=False
                   )


#%%

img.shape


img.ndim

#%%

import rioxarray

data = rioxarray.open_rasterio(filepath)
print(data)  # DataArray with bands as a dimension

#%%

data.data[11]# (bands, height, width)
#%%
import rasterio

with rasterio.open(filepath) as src:
    print(src.count)  # number of bands
    band4 = src.read(4)  # read band 4 (Red)
    all_bands = src.read()  # shape: (13, height, width)


#%%

all_bands.shape
#%%
def create_alteration_index(band1: str, band2: str) -> str:
    """Create a string representation of an alteration index formula.

    Args:
        band1: The name of the first spectral band.
        band2: The name of the second spectral band.

    Returns:
        A string representing the alteration index formula.
    """
    return f"({band1} - {band2}) / ({band1} + {band2})"



#%%

class Sentinel2Dataset(object):
    def __init__(self, img_folder: str):
        self.all_band_names = ("B01",
                                "B02",
                                "B03",
                                "B04",
                                "B05",
                                "B06",
                                "B07",
                                "B08",
                                "B8A",
                                "B09",
                                "B11",
                                "B12",
                                )
        self.img_folder = img_folder
        self.filepaths = glob(f"{self.img_folder}/*.tif")
        print("reading images from filepaths...")
        self.imgs = np.array([get_tiff_img(path=filepath,
                                           return_all_bands=True,
                                           normalize_bands=False
                                           )
                              for filepath in self.filepaths
                            ]
                            )
        print("completed reading images.")
        
        
    def cal_alteration_index(self, img):
        B11_index = self.all_band_names.index("B11")
        B12_index = self.all_band_names.index("B12")
        band11 = img[:, :, B11_index]
        band12 = img[:, :, B12_index]        
        self.alteration = (band11 / band12)
        return self.alteration
        
        
    def cal_ferric_oxide_index(self, img):
        B11_index = self.all_band_names.index("B11")
        B8_index = self.all_band_names.index("B08")
        band11 = img[:, :, B11_index]
        band8 = img[:, :, B8_index]
        self.ferric_oxide = band11 / band8
        return self.ferric_oxide
    
    
    def cal_ferrous_iron_index(self, img):
        B12_index = self.all_band_names.index("B12")
        B8_index = self.all_band_names.index("B08")
        B3_index = self.all_band_names.index("B03")
        B4_index = self.all_band_names.index("B04")

        band12 = img[:, :, B12_index]
        band8 = img[:, :, B8_index]
        band3 = img[:, :, B3_index]
        band4 = img[:, :, B4_index]
        
        band_x = band12 / band8
        band_y = band3 / band4
        self.ferrous_iron = band_x + band_y
        return self.ferrous_iron
    
    def cal_ferrous_silicates_index(self, img):
        B12_index = self.all_band_names.index("B12")
        B11_index = self.all_band_names.index("B11")
        band12 = img[:, :, B12_index]
        band11 = img[:, :, B11_index]        
        self.ferrous_silicates = band12 / band11
        return self.ferrous_silicates
        
    def cal_iron_oxide_index(self, img):
        B5_index = self.all_band_names.index("B05")
        B1_index = self.all_band_names.index("B01")
        band5 = img[:, :, B5_index]
        band1 = img[:, :, B1_index]        
        self.iron_oxide = band5 / band1
        return self.iron_oxide
    
    def cal_gossan_index(self, img):
        B11_index = self.all_band_names.index("B11")
        B4_index = self.all_band_names.index("B04")
        band4 = img[:, :, B4_index]
        band11 = img[:, :, B11_index]        
        self.gossan = band11 / band4
        return self.gossan
    


#%%


   
dataset = Sentinel2Dataset(img_folder=folder)

#%%

dataset.imgs.shape
# %%
sample_img = dataset.imgs[0]
# %%
alteration_index = dataset.cal_alteration_index(img=sample_img)

alteration_index.shape
# %%
alteration_index
# %%

sample_img[:, :, 12].shape
# %%

sample_img.shape

# %%
ferric_oxide =  dataset.cal_ferric_oxide_index(img=sample_img)



# %%
ferric_oxide.shape
# %%
gossan_index = dataset.cal_gossan_index(img=sample_img)
# %%
gossan_index.shape
# %%
np.dstack([alteration_index, ferric_oxide, gossan_index]).shape
# %%
indices_stack = np.dstack([alteration_index, ferric_oxide, gossan_index])
# %%

import matplotlib.pyplot as plt
import torch


#%%

image = torch.clamp(indices_stack / 10000, min=0, max=1).numpy()
        
#%% plot the image
fig, ax = plt.subplots()
ax.imshow(indices_stack)
# %%
