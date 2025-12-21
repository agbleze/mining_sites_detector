
import numpy as np
import rasterio
import torch
from torch import nn

def get_tiff_img(path, return_all_bands, bands=("B01", "B03", "B02"),
                 normalize_bands=True
                ):
    all_band_names = ("B01","B02", "B03","B04","B05", "B06",
                      "B07","B08","B8A","B09","B11","B12"
                    )
    if return_all_bands:
        band_indexs = [all_band_names.index(band_name) for band_name in all_band_names]
    
    else:
        band_indexs = [all_band_names.index(band_name) for band_name in bands]

    with rasterio.open(path) as src:
        img_bands = [src.read(band) for band in range(1,13)]
    dstacked_bands = np.dstack([img_bands[band_index] for band_index in band_indexs])
    if normalize_bands:
        # Normalize bands to 0-255
        dstacked_bands = ((dstacked_bands - dstacked_bands.min()) / 
                          (dstacked_bands.max() - dstacked_bands.min()) * 255
                          ).astype(np.uint8)

    return dstacked_bands



def compute_mean_var(dataloader):
    total_mean = 0
    total_var = 0
    total_count = 0
    for batch in dataloader:
        batch_mean = batch.mean(axis=(0,2,3))
        batch_var = batch.var(axis=(0,2,3))
        batch_count = batch.shape[0]
        total_count += batch_count
        batch_weight = float(batch_count) / total_count
        existing_weight = 1.0 - batch_weight
        new_total_mean = total_mean * existing_weight + batch_mean * batch_weight
        
        # The variance is computed using the lack-of-fit sum of squares
        total_var = (total_var + (total_mean - new_total_mean)**2 ) * existing_weight + \
                    (batch_var + (batch_mean - new_total_mean)**2) * batch_weight
        total_mean = new_total_mean
    return {"mean": total_mean, "var": total_var}
    
def normalize_batch(batch, mean, var):
    std = torch.max(torch.sqrt(var), torch.tensor(1e-8))
    return (batch - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
 
        
class PrestemNormTransform(nn.Module):
    
    def __init__(self, mean, var, **kwargs):
        super(PrestemNormTransform).__init__()
        self.mean = mean
        self.var = var
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)

    def forward(self, x):
        x = normalize_batch(batch=x, 
                            mean=self.mean, 
                            var=self.var
                            )
        return x
        