from sklearn.decomposition import PCA
from glob import glob
import numpy as np
from mining_sites_detector.import_utils import get_tiff_img



def compute_PC(img, components: int):
    H, W, C = img.shape
    X = img.reshape(-1, C)
    pca = PCA(n_components=components)
    pc = pca.fit_transform(X)
    pcimg = pc.reshape(H, W)
    return pcimg

class Sentinel2Dataset(object):
    def __init__(self, img_folder: str, load_on_initialize: bool = True):
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
        if load_on_initialize:
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
            
        
    def compute_alteration_index(self, img):
        band11 = self.get_band(band_name="B11", img=img)
        band12 = self.get_band(band_name="B12", img=img)      
        self.alteration = (band11 / band12)
        return self.alteration
        
        
    def compute_ferric_oxide_index(self, img):
        band11 = self.get_band(band_name="B11", img=img)
        band8 = self.get_band(band_name="B08", img=img)
        self.ferric_oxide = band11 / band8
        return self.ferric_oxide
    
    
    def compute_ferrous_iron_index(self, img):
        band12 = self.get_band(band_name="B12", img=img)
        band8 = self.get_band(band_name="B08", img=img)
        band3 = self.get_band(band_name="B03", img=img)
        band4 = self.get_band(band_name="B04", img=img)
        
        band_x = band12 / band8
        band_y = band3 / band4
        self.ferrous_iron = band_x + band_y
        return self.ferrous_iron
    
    def compute_ferrous_silicates_index(self, img):
        band12 = self.get_band(band_name="B12", img=img)
        band11 = self.get_band(band_name="B11", img=img)      
        self.ferrous_silicates = band12 / band11
        return self.ferrous_silicates
        
    def compute_iron_oxide_index(self, img):
        band5 = self.get_band(band_name="B05", img=img)
        band1 = self.get_band(band_name="B01", img=img)       
        self.iron_oxide = band5 / band1
        return self.iron_oxide
    
    def compute_gossan_index(self, img):
        band4 = self.get_band(band_name="B04", img=img)
        band11 = self.get_band(band_name="B11", img=img)       
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
        band12 = self.get_band(band_name="B12", img=img)
        band8 = self.get_band(band_name="B08", img=img)
        band11 = self.get_band(band_name="B11", img=img)  
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
        band4 = self.get_band(band_name="B04", img=img)
        band8 = self.get_band(band_name="B08", img=img)
        band11 = self.get_band(band_name="B11", img=img)  
        self.three_band_urband_index =  band4 + band11 - band8  
        return self.three_band_urband_index
    
    
    def compute_BAEI(self, img):
        """
        Compute the Built-Up Area Extraction Index (BAEI) for the given image.
        The Built-Up Area Extraction Index (BAEI) is designed to enhance the detection 
        of built-up areas using three spectral bands: Red, NIR, and SWIR1.
        
        Equation is cited from: 
        """
        band4 = self.get_band(band_name="B04", img=img)
        band3 = self.get_band(band_name="B03", img=img)
        band11 = self.get_band(band_name="B11", img=img)   
        baei_numerator = band4 + 0.3
        baei_denominator = band3 + band11
        self.built_up_area_extraction_index = baei_numerator / baei_denominator
        return self.built_up_area_extraction_index
    
    def compute_BBI(self, img):
        """
        Built-up and Bare Land Index
        
        Equation is cited from: 
        """
        band4 = self.get_band(band_name="B04", img=img)
        band2 = self.get_band(band_name="B02", img=img)
        band3 = self.get_band(band_name="B03", img=img) 
        leftside_bbi_numerator = band2 - band3
        leftside_bbi_denominator = band2 + band3
        rightside_bbi_numerator = band4 - band3
        rightside_bbi_denominator = band4 + band3
        leftside_bbi = leftside_bbi_numerator / leftside_bbi_denominator
        rightside_bbi = rightside_bbi_numerator / rightside_bbi_denominator
        self.built_up_bare_land_index = leftside_bbi + rightside_bbi
        return self.built_up_bare_land_index

    
    def compute_Brightness_index(self, img):
        """
        BI_Br BrightnessIndex
        
        """
        band4 = self.get_band(band_name="B04", img=img)
        band8 = self.get_band(band_name="B08", img=img)
        leftside = band4**2
        rightside = band8**2
        self.brightness_index = np.sqrt(leftside + rightside)
        return self.brightness_index
    
    def compute_BLFEI(self, img):
        """
        BLFEI Built-up Land Features Extraction Index
        """
        
        b3_index = self.all_band_names.index("B03")
        b4_index = self.all_band_names.index("B04")
        b12_index = self.all_band_names.index("B12")
        b11_index = self.all_band_names.index("B11")
        band3 = img[:, :, b3_index]
        band4 = img[:, :, b4_index]
        band12 = img[:, :, b12_index]
        band11 = img[:, :, b11_index]
        bands_avg = (band3 + band4 + band12) / 3
        self.blfei = (bands_avg - band11) / (bands_avg + band11)
        
    def compute_BRBA(self, img):
        """
        BRBA Band Ratio for Built-up Area
        
        """
        band3 = self.get_band(band_name="B03", img=img)
        band8 = self.get_band(band_name="B08", img=img)
        self.BRBA = band3 / band8
        return self.BRBA
        
    def compute_bare_soil_index(self, img):
        """
        BSI
        Bare Soil Index
        
        :param self: Description
        :param img: Description
        """
        band12 = self.get_band(band_name="B12", img=img)
        band4 = self.get_band(band_name="B04", img=img)
        band8 = self.get_band(band_name="B08", img=img)
        band2 = self.get_band(band_name="B02", img=img)
        
        leftside_numerator = band12 + band4
        rightside_numerator = band8 - band2
        leftside_denominator = band12 + band4
        rightside_denominator = band8 + band2
        numerator = leftside_numerator - rightside_numerator
        denominator = leftside_denominator + rightside_denominator
        self.bare_soil_index = numerator / denominator
        return self.bare_soil_index
        
    
    def compute_crust_index(self, img):
        band4 = self.get_band(band_name="B04", img=img)
        band2 = self.get_band(band_name="B02", img=img)
        numerator = (band4 - band2)
        denominator = (band4 + band2)
        res = numerator / denominator
        self.crust_index = 1 - res
        return self.crust_index
    
    def compute_new_builtup_index(self, img):
        """
        
        NBI
        New Built-up Index
        """
        band4 = self.get_band(band_name="B04", img=img)
        band11 = self.get_band(band_name="B11", img=img)
        band8 = self.get_band(band_name="B08", img=img)
        numerator = band4 * band11
        self.new_builtup_index = numerator / band8
        return self.new_builtup_index
    
    def compute_builtup_index(self, img):
        """
        
        BU
        Built-up Index NDBI—NDVI
        """
        band4 = self.get_band(band_name="B04", img=img)
        band8 = self.get_band(band_name="B08", img=img)
        band11 = self.get_band(band_name="B11", img=img)
        ndvi_numerator = band8 - band4
        ndvi_denominator = band8 + band4
        ndvi = ndvi_numerator / ndvi_denominator
        
        ndbi_numerator = band11 - band8
        ndbi_denominator = band11 + band8
        ndbi = ndbi_numerator / ndbi_denominator
        self.builtup_index = ndbi - ndvi
        return self.builtup_index
    
    def compute_CBI(self, img):
        """
        CBI
        Combinational Build-up Index
        
        """
        band3 = self.get_band(band_name="B03", img=img)
        band8 = self.get_band(band_name="B08", img=img)
        band4 = self.get_band(band_name="B04", img=img)
        ndwi_numerator = band3 - band8
        ndwi_denominator = band3 + band8
        ndwi = ndwi_numerator / ndwi_denominator
        savi_numerator = band8 - band4
        savi_denominator = band8 + band4 + 0.5
        savi = (savi_numerator / savi_denominator) * 1.5
        pc1 = compute_PC(img, components=1)
        cbi_leftside = (pc1 + ndwi) / 2
        cbi_numerator = cbi_leftside - savi
        cbi_denominator = cbi_leftside + savi
        self.cbi = cbi_numerator / cbi_denominator
        return self.cbi
    
    def compute_dry_bareness_index(self, img):
        """
        DBSI
        Dry Bareness Index
        """  
        band4 = self.get_band(band_name="B04", img=img)
        band8 = self.get_band(band_name="B08", img=img)
        band3 = self.get_band(band_name="B03", img=img)
        band11 = self.get_band(band_name="B11", img=img)
        
        ndvi_numerator = band8 - band4
        ndvi_denominator = band8 + band4
        ndvi = ndvi_numerator / ndvi_denominator
        dbsi_leftside = (band11 - band3) / (band11 + band3)
        self.dry_bareness_index = dbsi_leftside - ndvi
        return self.dry_bareness_index
          
    def get_band(self, band_name: str, img):
        if band_name not in self.all_band_names:
            raise ValueError(f"{band_name} is not a valid band. Valid bands are: {self.all_band_names}")
        
        band_index = self.all_band_names.index(band_name)
        band = img[:, :, band_index]
        return band + 1e-5
        
    