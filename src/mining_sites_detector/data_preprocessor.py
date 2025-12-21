
import os
import matplotlib.pyplot as plt
import torch
from torchgeo.datasets import stack_samples
from rasterio import plot
from typing import Union, Callable, Any, Optional, Tuple, List, Dict, cast
from pathlib import Path
import os
import numpy as np
import pandas as pd
from torch import Tensor
import torch
from torchgeo.datasets import stack_samples
from torchgeo.datasets.geo import NonGeoClassificationDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import torch
from mining_sites_detector.index import Sentinel2Dataset
from mining_sites_detector.import_utils import get_tiff_img


class MineSiteImageFolder(Dataset):
    def __init__(self, root: Union[str, Path],
                 loader: Callable[[str], Any],
                 target_file_path: str,
                 extensions: Optional[Tuple[str, ...]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 allow_empty: bool = False,
                 class_to_idx: Optional[Union[Dict, None]] = None,
                 fetch_for_all_classes = True,
                 target_file_has_header = False,
                 img_name: Optional[Union[str, None]] = None,
                 img_idx: Optional[Union[int, None]] = None,
                 return_all_bands=False,
                 bands=("B04", "B03", "B02"),
                 normalize_bands=True
                 ) -> None:
        self.root = root
        self.target_file_path = target_file_path
        self.target_file_has_header = target_file_has_header
        self.fetch_for_all_classes = fetch_for_all_classes
        self.img_name = img_name
        self.img_idx = img_idx
        self.loader = loader
        self.target_transform = target_transform
        self.transform = transform
        #self.is_valid_file = is_valid_file
        self.allow_empty = allow_empty
        self.return_all_bands = return_all_bands
        self.bands = bands
        self.normalize_bands = normalize_bands
          
        if not class_to_idx:
            self.classes, self.class_to_idx = self.find_classes()
            class_to_idx = self.class_to_idx
        samples = self.make_dataset(root=self.root,
                                    target_file_path=self.target_file_path,
                                    class_to_idx=self.class_to_idx,
                                    img_name=self.img_name, img_idx=self.img_idx,
                                    target_file_has_header=self.target_file_has_header,
                                    )  
        self.samples = samples
        
 
    def find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """This returns a hard-coded binary classes and class to index.
            It is hardcorded because the classes and not provided in the file 
            but deduced from their binary nature and task at hand
        """
        
        classes = ["not_mining_site", "mining_site"]
        class_to_idx = {"not_mining_site": 0, "mining_site": 1}
        return classes, class_to_idx
    
    def make_dataset(self, root, 
                     target_file_path: Union[str, Path],
                     class_to_idx: Optional[Dict],
                     img_name: Optional[str], 
                     img_idx: Optional[int], fetch_for_all_classes: bool = True,
                     target_file_has_header = False,
                     
                     ):
        """Generate a list of samples of each class of the form (path_to_sample, class)

        Args:
            root (_type_): _description_
            target_file_path (Union[str, Path]): Csv file with first column as file name and second column as target
                                                If file does not have a header then set target_file_has_header = False
            img_name (_type_): _description_
            img_idx (_type_): _description_
            target_file_has_header (bool): _description_ = False

        Raises:
            ValueError: _description_
        """
        
        if not root:
            root = self.root
            
        if not class_to_idx:
            classes, class_to_idx = self.find_classes()
            
        if not target_file_has_header:
            target_file_df = pd.read_csv(target_file_path, header=None)
        else:
            target_file_df = pd.read_csv(target_file_path)          
        
        
        instances = []
        if fetch_for_all_classes:
            cls_idx = class_to_idx.values()
            columns = target_file_df.columns.to_list()
            if len(columns) > 2:
                target_file_df = target_file_df.drop(columns[0], axis=1)
                columns = columns[1:]
            
            cls_target_df = target_file_df[target_file_df[columns[1]].isin(cls_idx)]
            img_names, img_target_idx = cls_target_df[columns[0]].values, cls_target_df[columns[1]].values
            img_path = [os.path.join(root, img_nm) for img_nm in img_names]
            
            for data_sample in zip(img_path, img_target_idx):
                instances.append(data_sample)
        else: 
            if not img_name and not img_idx:
                raise ValueError(f"""img_name or img_idx must be provided if you want to 
                                    fetch sample for a particular image. Or set fetch_for_all_classes: bool = True
                                 
                                 """)
            
            if not img_name:
                img_name = os.listdir(root)[img_idx]
            
            img_path = os.path.join(root, img_name)
            img_target_idx = target_file_df[target_file_df[0]==img_name][1].values[0]
            instances.append((img_path, img_target_idx))
            
        return instances
        
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path=path,
                             return_all_bands=self.return_all_bands,
                             bands=self.bands,
                             normalize_bands=self.normalize_bands
                            )
        #sample_dstack = np.dstack([band for band in sample])
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
            
        return sample, target
    
    def __len__(self) -> int:
        return len(self.samples)

class NonGeoMineSiteClassificationDataset(MineSiteImageFolder):
    def __init__(self, root: str,
                 loader,
                 target_file_path: str,
                 is_valid_file = None,
                 transforms = None,
                 extensions: Optional[Tuple[str, ...]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 allow_empty: bool = False,
                 class_to_idx: Optional[Union[Dict, None]] = None,
                 fetch_for_all_classes = True, 
                 target_file_has_header = False,
                 img_name: Optional[Union[str, None]] = None,
                 img_idx: Optional[Union[int, None]] = None,
                 return_all_bands=False,
                 bands=("B04", "B03", "B02"),
                 normalize_bands=True,
                 create_indices = True
                 ) -> None:
        self.root = root
        self.loader = loader
        self.target_file_path = target_file_path
        self.target_file_has_header = target_file_has_header
        self.transforms = transforms
        self.transform = transform
        self.return_all_bands = return_all_bands
        self.bands = bands
        self.normalize_bands = normalize_bands
        self.create_indices = create_indices
        super().__init__(
                        root=root,
                        is_valid_file=is_valid_file,
                        loader=loader, class_to_idx=class_to_idx,
                        target_file_path=target_file_path,
                        target_file_has_header=target_file_has_header,
                        transform=transform,
                        fetch_for_all_classes=fetch_for_all_classes,
                        img_name=img_name, img_idx=img_idx,
                        target_transform=target_transform, allow_empty=allow_empty,
                        return_all_bands=self.return_all_bands, bands=self.bands,
                        normalize_bands=self.normalize_bands
                        
                    )
        
        self.mnfolder = MineSiteImageFolder(root=self.root, target_file_path=self.target_file_path,
                                            target_file_has_header=self.target_file_has_header,
                                            loader=self.loader,return_all_bands=self.return_all_bands, 
                                            bands=self.bands,
                                            normalize_bands=self.normalize_bands
                                            )
        
    def _load_image(self, index) -> Tuple[Tensor, Tensor]: # take-out, import NonGeoClassificationDataset and use to override there
        """Load a single image with its class label as tensor

        Args:
            index (_type_): _description_
            
        """
        self.img, self.label = self.mnfolder[index]
        # create indices and stack
        _created_indices = False
        if self.create_indices:
            indices_creater = Sentinel2Dataset(img_folder=self.root, load_on_initialize=False)
            feoxide = indices_creater.compute_ferric_oxide_index(img=self.img)
            mbsi = indices_creater.compute_modified_baresoil_index(self.img)
            brightness_index = indices_creater.compute_Brightness_index(self.img)
            alteration_index = indices_creater.compute_alteration_index(self.img)
            builtup_index = indices_creater.compute_builtup_index(self.img)
            fesil = indices_creater.compute_ferrous_silicates_index(self.img)
            gossan_index = indices_creater.compute_gossan_index(self.img)
            ironoxide = indices_creater.compute_iron_oxide_index(self.img)
            dry_bareness = indices_creater.compute_dry_bareness_index(self.img)
            crust = indices_creater.compute_crust_index(self.img)
            bbi = indices_creater.compute_BBI(self.img)
            soil_index = indices_creater.compute_bare_soil_index(img=self.img)
            brba = indices_creater.compute_BRBA(self.img)
            #baei = indices_creater.compute_BAEI(self.img)
            
            indices = np.dstack([feoxide, brightness_index, alteration_index, builtup_index, fesil,
                                gossan_index, ironoxide, dry_bareness, crust, bbi, soil_index, brba,
                                #baei
                                ]
                                )
            #print(f"shape indices:{indices.shape}")
            
            _created_indices= True
            
        # permute indices
        if _created_indices:
            indices_tensor = torch.tensor(indices).permute(2,0,1).float()
        else:
            indices_tensor = None
        img_tensor = torch.tensor(self.img).permute(2,0,1).float()
        label_tensor = torch.tensor(self.label).float()
        return img_tensor, label_tensor, indices_tensor
    
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image, label, indice = self._load_image(index)
        sample = {"image": image, "label": label, "indice": indice}
        
        if self.transforms:
            sample = self.transforms(sample)
        return sample
        


class MineSiteDataset(NonGeoMineSiteClassificationDataset):

    all_band_names = ("B01",
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
    
    rgb_bands = ("B04", "B03", "B02")
    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}
    separate_files = False
    
    def __init__(self, root: Union[str, Path],
                 loader: Callable[[str], Any],
                 target_file_path: str,
                 extensions: Optional[Tuple[str, ...]] = None,
                 transforms: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 allow_empty: bool = False,
                 #class_to_idx: Optional[Union[Dict, None]] = None,
                 fetch_for_all_classes = True, 
                 target_file_has_header = False,
                 img_name: Optional[Union[str, None]] = None,
                 img_idx: Optional[Union[int, None]] = None,
                 #bands = BAND_SETS["all"],
                 class_to_idx = None,
                 return_all_bands=False, 
                 bands=("B04", "B03", "B02"),
                normalize_bands=True,
                create_indices: bool = False
                 #index=1
                ):
        self.root = root
        self.target_file_path = target_file_path
        self.target_file_has_header = target_file_has_header
        self.fetch_for_all_classes = fetch_for_all_classes
        self.img_name = img_name
        self.img_idx = img_idx
        self.loader = loader
        self.transforms = transforms
        self.bands = bands
        self.class_to_idx = class_to_idx
        self.band_indices = Tensor([self.all_band_names.index(b) for b in self.all_band_names]).long()
        self.return_all_bands = return_all_bands
        self.normalize_bands = normalize_bands
        self.create_indices = create_indices

        super().__init__(root=self.root, target_file_path= self.target_file_path,
                         target_file_has_header=self.target_file_has_header,
                        fetch_for_all_classes=self.fetch_for_all_classes,
                        loader=self.loader, class_to_idx=self.class_to_idx,
                        is_valid_file=is_valid_file, transform=transforms,
                        return_all_bands=return_all_bands,
                        bands=bands,
                        normalize_bands=self.normalize_bands,
                        create_indices=self.create_indices
                        )
    
    
    
    def __getitem__(self,index):
        img, label, indice = self._load_image(index)
        self.sample = {"image": img, "label": label, "indice": indice}
        if self.transforms:
            self.sample = self.transforms(self.sample)

        return self.sample
    def plot(self, index):
        self._load_image(index)
        plt.imshow(self.img)
        plt.show()
        
    
    def _plot(self, sample, show_title):
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError(f"band {band} not found in existing bands which are {self.bands}")
            
        image = np.take(sample["image"].numpy(), indices=rgb_indices, axis=0)
        #rgb_bands = np.dstack([image[0], image[1], image[2]])
        #image = ((rgb_bands - rgb_bands.min()) / (rgb_bands.max() - rgb_bands.min()) * 255).astype(np.uint8)
        image = np.rollaxis(image, 0, 3)
        #image = np.clip(image / 3000, 0, 1)   
        
        label = cast(int, sample["label"].item())
        label_class = self.classes[label]
        
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(image)
        ax.axis("off")
        
        if show_title:
            title = f"Label: {label_class}"
            ax.set_title(title)
            
        return fig




def get_data(train_img_dir, val_image_dir, 
             train_target_file_path: str,
             val_target_file_path: str,
             target_file_has_header=True, 
             loader=get_tiff_img,
             return_all_bands=True,
             batch_size=10, drop_last=True,
             num_workers=4,
             transforms: Union[Callable, None]=None,
             normalize_bands=True, 
             create_indices: bool = False
            ):
    train_dataset = MineSiteDataset(root=train_img_dir, 
                                    target_file_path=train_target_file_path,
                                    target_file_has_header=target_file_has_header, 
                                    loader=loader,
                                    return_all_bands=return_all_bands,
                                    transforms=transforms,
                                    normalize_bands=normalize_bands,
                                    create_indices=create_indices
                                    
                                    )
    val_dataset = MineSiteDataset(root=val_image_dir, 
                                  target_file_path=val_target_file_path,
                                    target_file_has_header=target_file_has_header, 
                                    loader=loader,
                                    return_all_bands=return_all_bands,
                                    transforms=transforms, normalize_bands=normalize_bands,
                                    create_indices=create_indices
                                    
                                    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  collate_fn=stack_samples, drop_last=drop_last
                                  )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                collate_fn=stack_samples, drop_last=drop_last
                                )
    
    return train_dataloader, val_dataloader
    
