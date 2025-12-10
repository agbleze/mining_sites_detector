
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
    
    def compute_modified_baresoil_index(self, img):
        """
        Compute the Modified Baresoil Index (MBI) for the given image.
        Modified Bare Soil Index (MBI) for Sentinel-2 can be 
        computed using the SWIR1, SWIR2, and NIR spectral bands to improve 
        the discrimination between bare soil and built-up areas.
        
        Help reduce the misclassification of built-up areas as bare soil, a common issue with the standard BSI. 
        The resulting values for bare soil are typically positive and higher than for other land
        
        """
        B12_index = self.all_band_names.index("B12")
        B8_index = self.all_band_names.index("B08")
        B11_index = self.all_band_names.index("B11")
        band12 = img[:, :, B12_index]
        band8 = img[:, :, B8_index]
        band11 = img[:, :, B11_index]   
        mbi_numerator = band11 - band12 - band8
        mbi_denominator = band11 + band12 + band8     
        self.modified_baresoil = (mbi_numerator / mbi_denominator) + 0.5
        return self.modified_baresoil
    
    def compute_3BUI_index(self, img):
        """
        Compute the 3-Band Built-Up Index (3BUI) for the given image.
        The 3-Band Built-Up Index (3BUI) is designed to enhance the detection 
        of built-up areas using three spectral bands: Red, NIR, and SWIR1.
        
        Equation is cited from: 
        """
        B4_index = self.all_band_names.index("B04")
        B8_index = self.all_band_names.index("B08")
        B11_index = self.all_band_names.index("B11")
        band4 = img[:, :, B4_index]
        band8 = img[:, :, B8_index]
        band11 = img[:, :, B11_index]   
        self.three_band_urband_index =  band4 + band11 - band8  
        return self.three_band_urband_index
    
    
    def compute_BAEI(self, img):
        """
        Compute the Built-Up Area Extraction Index (BAEI) for the given image.
        The Built-Up Area Extraction Index (BAEI) is designed to enhance the detection 
        of built-up areas using three spectral bands: Red, NIR, and SWIR1.
        
        Equation is cited from: 
        """
        B4_index = self.all_band_names.index("B04")
        B3_index = self.all_band_names.index("B03")
        B11_index = self.all_band_names.index("B11")
        band4 = img[:, :, B4_index]
        band3 = img[:, :, B3_index]
        band11 = img[:, :, B11_index]   
        baei_numerator = band4 + 0.3
        baei_denominator = band3 + band11
        self.built_up_area_extraction_index = baei_numerator / baei_denominator
        return self.built_up_area_extraction_index
    
    def compute_BBI(self, img):
        """
        Built-up and Bare Land Index
        
        Equation is cited from: 
        """
        B2_index = self.all_band_names.index("B02")
        B3_index = self.all_band_names.index("B03")
        B4_index = self.all_band_names.index("B04")
        
        band4 = img[:, :, B4_index]
        band2 = img[:, :, B2_index]
        band3 = img[:, :, B3_index]   
        leftside_bbi_numerator = band2 - band3
        leftside_bbi_denominator = band2 + band3
        rightside_bbi_numerator = band4 - band3
        rightside_bbi_denominator = band4 + band3
        leftside_bbi = leftside_bbi_numerator / leftside_bbi_denominator
        rightside_bbi = rightside_bbi_numerator / rightside_bbi_denominator
        self.built_up_bare_land_index = leftside_bbi + rightside_bbi
        return self.built_up_bare_land_index

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
sample_img.dtype
# %%
indices_stack.max()
# %%
gossan_index.max()
# %%
ferric_oxide.max()
# %%
iron_oxide = dataset.cal_iron_oxide_index(img=sample_img)
# %%
iron_oxide.max()
# %%
from sklearn.preprocessing import StandardScaler
# %%
scaler = StandardScaler()
# %%
scaled_indices = scaler.fit_transform(indices_stack.reshape(-1, indices_stack.shape[-1]))
# %%
scaled_indices.min()
# %%
#import torch.functional as F
import torch.nn.functional as F
# %%
F.normalize(torch.tensor(indices_stack), dim=0).numpy().min()
# %%
from torchvision import transforms
# %%
transforms.Normalize()

# %%
import numpy as np

def compute_mean_std_numpy(arrs):
    # arrs: list of arrays shaped (bands, H, W)
    stacked = np.concatenate([a.reshape(a.shape[0], -1) for a in arrs], axis=1)
    mean = stacked.mean(axis=1)
    std  = stacked.std(axis=1)
    return mean, std

