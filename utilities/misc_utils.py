"""
Copyright 2020 The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#Approved for public release, 20-563

import sys
sys.path.append("..")

from glob import glob
import os
import numpy as np
import gdal
import json
from keras import backend as K
from keras.applications import imagenet_utils

def no_nan_mse(y_true, y_pred, ignore_value=-10000):
    mask_true = K.cast(K.not_equal(y_true, ignore_value), K.floatx())
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
    return masked_mse

def get_checkpoint_dir(args):
    height_str = "with_height" if args.add_height else "without_height"
    aug_str = "with_aug" if args.augmentation else "without_aug"
    checkpoint_sub_dir = height_str + "_" + aug_str
    
    checkpoint_dir = os.path.join(args.checkpoint_dir, checkpoint_sub_dir)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    return checkpoint_dir,checkpoint_sub_dir


def load_vflow(vflow_path, agl):
    
    vflow_data = json.load(open(vflow_path, "r"))
    
    mag = agl * vflow_data["scale"]
    
    xdir,ydir = np.sin(vflow_data["angle"]),np.cos(vflow_data["angle"])

    vflow = np.zeros((agl.shape[0],agl.shape[1],2))
    vflow[:,:,0] = mag * xdir
    vflow[:,:,1] = mag * ydir
    
    vflow_info = json.load(open(vflow_path, "r"))
    
    return vflow,mag,xdir,ydir,vflow_data["angle"]


def get_data(args, is_train=True, rgb_paths_only=False):
    split_dir = args.train_sub_dir if is_train else args.test_sub_dir
    rgb_paths = glob(os.path.join(args.dataset_dir, split_dir, "*_RGB*.tif"))
    if rgb_paths_only:
        return rgb_paths
    vflow_paths = [rgb_path.replace("_RGB", "_VFLOW").replace(".tif", ".json") for rgb_path in rgb_paths]
    agl_paths = [rgb_path.replace("_RGB", "_AGL") for rgb_path in rgb_paths]
    data = [(rgb_paths[i], vflow_paths[i], agl_paths[i]) for i in range(len(rgb_paths))]
    return data
    
def load_image(image_path):
    image = gdal.Open(image_path)
    image = image.ReadAsArray()
    if len(image.shape)==3:
        image = np.transpose(image, [1,2,0])
    return image

def save_image(image, out_path):
    driver = gdal.GetDriverByName('GTiff')
    if len(image.shape)==2:
        out_channels = 1
    else:
        out_channels = image.shape[2]
    dataset = driver.Create(out_path, image.shape[1], image.shape[0], out_channels, gdal.GDT_Float32)
    if len(image.shape)==2:
        dataset.GetRasterBand(1).WriteArray(image)
    else:
        for c in range(out_channels):
            dataset.GetRasterBand(c+1).WriteArray(image[:,:,c])
            
    dataset.FlushCache()

def image_preprocess(image_batch):
    return imagenet_utils.preprocess_input(image_batch) / 255.0

def get_batch_inds(idx, batch_sz):
    N = len(idx)
    batch_inds = []
    idx0 = 0
    to_process = True
    while to_process:
        idx1 = idx0 + batch_sz
        if idx1 > N:
            idx1 = N
            idx0 = idx1 - batch_sz
            to_process = False
        batch_inds.append(idx[idx0:idx1])
        idx0 = idx1
    return batch_inds
