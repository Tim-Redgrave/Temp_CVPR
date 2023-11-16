import os
import json
import numpy as np
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.datapoints as datapoints
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
import torch.utils.data as data_utl
from PIL import Image
from torchvision.io import read_image
import sys

def bounding_boxes_to_mask(bounding_boxes,image_size,boolean_mask=False):
    mask = torch.zeros((1,image_size,image_size))
    for bounding_box in bounding_boxes:
        mask = transforms.functional.erase(mask,i=bounding_box[1],j=bounding_box[0],w=bounding_box[2],h=bounding_box[3],v=1.)
    if boolean_mask:
        mask = mask.to(torch.bool)
    return mask

class SIP_Dataset(Dataset):
    def __init__(self, data_path, annotations_file, image_size, transform=None,salience_form=None, unannotatied_split=None,default_salience_map_path="zeros_img.png"):
        self.img_labels = pd.read_csv(annotations_file)
        if unannotatied_split is not None:
            rand_sample_indices = list(self.img_labels.sample(frac=unannotatied_split).index.values)
            if salience_form is not None and salience_form == "salience_maps":
                self.img_labels.loc[rand_sample_indices,"salience_map_path"] = default_salience_map_path
            elif salience_form is not None and salience_form == "grad-cam_salience_maps":
                self.img_labels.loc[rand_sample_indices,"grad-cam_salience_maps"] = default_salience_map_path
            elif salience_form is not None and salience_form == "bounding_boxes":
                self.img_labels.loc[rand_sample_indices,"bounding_boxes"] = "[[0,0,0,0]]"
        self.data_path = data_path
        self.resize = transforms.Resize((image_size,image_size),antialias=True)
        self.bounding_box_resize = transforms.Resize((image_size,image_size),interpolation=transforms.InterpolationMode.NEAREST,antialias=True)
        self.transform = transform
        self.salience_form = salience_form
        self.image_size = image_size
        self.class_labels = self.img_labels["image_class"].values
        self.image_paths = self.img_labels["image_path"].values
        for i,image_path in enumerate(self.image_paths):
            self.image_paths[i] = os.path.join(data_path,image_path)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self,idx):
        #label = int(self.img_labels.loc[idx, "image_class"])
        label = self.class_labels[idx]
        #img_path = os.path.join(self.data_path, self.img_labels.loc[idx, "image_path"])
        img_path = self.image_paths[idx]
        image = read_image(img_path,torchvision.io.ImageReadMode.RGB)
        image = transforms.functional.convert_image_dtype(image,torch.float32)
        original_img_size = image.shape[-2:]
        image = self.resize(image)
        if self.salience_form is not None:
            if self.salience_form == "salience_maps":
                salience_map_path = os.path.join(self.data_path, self.img_labels.loc[idx, "salience_map_path"])
                salience_map = read_image(salience_map_path,torchvision.io.ImageReadMode.GRAY)
                salience_map = transforms.functional.convert_image_dtype(salience_map,torch.float32)
                salience_map = self.resize(salience_map)
                return image,label,salience_map
            elif self.salience_form == "grad-cam_salience_maps":
                salience_map_path = os.path.join(self.data_path, self.img_labels.loc[idx, "grad-cam_salience_maps"])
                salience_map = read_image(salience_map_path,torchvision.io.ImageReadMode.GRAY)
                salience_map = transforms.functional.convert_image_dtype(salience_map,torch.float32)
                salience_map = self.resize(salience_map)
                return image,label,salience_map
            elif self.salience_form == "bounding_boxes":
                bounding_boxes = json.loads(self.img_labels.loc[idx, "bounding_boxes"])
                bounding_boxes = [self.bounding_box_resize(datapoints.BoundingBox(bounding_box,format=datapoints.BoundingBoxFormat.XYWH,spatial_size=original_img_size)) for bounding_box in bounding_boxes]
                salience_map = bounding_boxes_to_mask(bounding_boxes,self.image_size)
                return image,label,salience_map
        else:
            return image,label

    
def create_SIP_DataLoader(data_path, annotations_file, image_size, transform=None,salience_form=None, val_split=None, unannotatied_split=None, default_salience_map_path="zeros_img.png", **kwargs):
    dataset_to_load = SIP_Dataset(data_path,annotations_file,image_size,transform,salience_form,unannotatied_split,default_salience_map_path)
    if val_split is not None:
        train_idx, validation_idx = train_test_split(np.arange(len(dataset_to_load)),test_size=val_split, shuffle=True,stratify=dataset_to_load.class_labels)
        train_dataset = torch.utils.data.Subset(dataset_to_load, train_idx)
        val_dataset = torch.utils.data.Subset(dataset_to_load, validation_idx)
        train_loader = torch.utils.data.DataLoader(train_dataset,**kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset,**kwargs)
        #train_data,val_data = torch.utils.data.random_split(dataset_to_load,[1-val_split,val_split])
        #train_loader = torch.utils.data.DataLoader(train_data,**kwargs)
        #val_loader = torch.utils.data.DataLoader(val_data,**kwargs)
        return train_loader,val_loader
    else:
        SIP_loader = torch.utils.data.DataLoader(dataset_to_load,**kwargs)
        return SIP_loader