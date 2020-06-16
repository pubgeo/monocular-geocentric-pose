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

import os
import numpy as np
from tqdm import tqdm
from glob import glob
from keras.layers import Input
from keras.models import load_model
from keras.callbacks import TensorBoard,ModelCheckpoint
import json
import cv2
import multiprocessing

from segmentation_models import UnetFlow
from utilities.misc_utils import get_data,get_batch_inds,load_image,save_image,no_nan_mse,image_preprocess,load_vflow,get_checkpoint_dir
from utilities.augmentation import augment,rotate_image,rotate_xydir

ignore_value = -10000

def train(args):
    train_data = get_data(args, is_train=True)
    val_data = get_data(args, is_train=False, rgb_paths_only=True)
    train_datagen, val_datagen, model = build_model(args, train_data, val_data)
    
    checkpoint_dir,_ = get_checkpoint_dir(args)
    
    checkpoint_filepath = os.path.join(checkpoint_dir, "weights.{epoch:02d}.hdf5")
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor="loss", verbose=0, save_best_only=False,
                                 save_weights_only=False, mode="auto", period=args.save_period)
    
    tensorboard = TensorBoard(log_dir=args.tensorboard_dir, write_graph=False)
    
    callbacks_list = [checkpoint, tensorboard]
    
    model.fit_generator(generator=train_datagen,
                        steps_per_epoch=(len(train_data) / args.batch_size + 1),
                        epochs=args.num_epochs, 
                        callbacks=callbacks_list)
    
def test(args):
    rgb_paths = get_data(args, is_train=False, rgb_paths_only=True)
    
    sub_dir = None
    if args.test_model_file is not None:
        weights_path = args.test_model_file
    else:
        checkpoint_dir,sub_dir = get_checkpoint_dir(args)
        weights_paths = glob(os.path.join(checkpoint_dir, "*.hdf5"))
        nums = [np.int(path.split(".")[-2]) for path in weights_paths]
        weights_path = weights_paths[np.argsort(nums)[-1]]
        
    model = load_model(weights_path, custom_objects={"no_nan_mse":no_nan_mse})
    
    predictions_dir = args.predictions_dir if sub_dir is None else os.path.join(args.predictions_dir, sub_dir)
    if not os.path.isdir(predictions_dir):
        os.makedirs(predictions_dir)
       
    
    angles = [angle for angle in range(0,360,36)] if args.test_rotations else [0]
    
    for rgb_path in tqdm(rgb_paths):
        basename = os.path.basename(rgb_path).replace("_RGB_", "_").replace(".tif", "")
        image = load_image(rgb_path)
        
        for angle in angles:
            
            out_dir_path = os.path.join(predictions_dir, basename + "_angle_%d_DIRPRED.json" % angle)
            out_mag_path = os.path.join(predictions_dir, basename + "_angle_%d_MAGPRED.tif" % angle)
            out_agl_path = os.path.join(predictions_dir, basename + "_angle_%d_AGLPRED.tif" % angle)
            
            image_rotated = rotate_image(np.copy(image), None, None, angle, image_only=True)
            image_rotated = image_preprocess(image_rotated)
            pred = model.predict(np.expand_dims(image_rotated, axis=0))
            agl = None
            
            if len(pred)==2:
                xydir,mag = pred
            else:
                xydir,mag,agl = pred

            xydir = xydir[0,:].tolist()
            mag = mag[0,:,:,0]

            json.dump(xydir, open(out_dir_path, "w"))
            save_image(mag, out_mag_path)
            if agl is not None:
                agl = agl[0,:,:,0]
                save_image(agl, out_agl_path)
                
    
def get_current_metrics(item):
    vflow_gt_path,agl_gt_path,dirpred_path,magpred_path,aglpred_path, angle = item
    dir_pred = json.load(open(dirpred_path, "r"))
    mag_pred = load_image(magpred_path)
    agl_pred = None if not os.path.isfile(aglpred_path) else load_image(aglpred_path)

    agl_gt = load_image(agl_gt_path)
    
    vflow_gt,mag_gt,xdir_gt,ydir_gt,_ = load_vflow(vflow_gt_path, agl_gt)
    
    if angle != 0:
        _,mag_gt,agl_gt = rotate_image(None, mag_gt, agl_gt, angle, image_only=False)
        xdir_gt,ydir_gt = rotate_xydir(xdir_gt, ydir_gt, angle)

    dir_gt = np.array([xdir_gt,ydir_gt])
    dir_gt /= np.linalg.norm(dir_gt)
    
    
    vflow_gt = cv2.merge((mag_gt*dir_gt[0], mag_gt*dir_gt[1]))

    dir_pred /= np.linalg.norm(dir_pred)

    cos_ang = np.dot(dir_pred, dir_gt)
    sin_ang = np.linalg.norm(np.cross(dir_pred,dir_gt))
    rad_diff = np.arctan2(sin_ang, cos_ang)
    angle_error = np.degrees(rad_diff)

    vflow_pred = cv2.merge((mag_pred*dir_pred[0], mag_pred*dir_pred[1]))

    mag_error = np.nanmean(np.abs(mag_pred-mag_gt))
    epe = np.nanmean(np.sqrt(np.sum(np.square(vflow_gt-vflow_pred), axis=2)))
    agl_error = None if agl_pred is None else np.nanmean(np.abs(agl_pred-agl_gt))
    
    return angle_error,mag_error,epe,agl_error
    
def metrics(args):
    predictions_dir = args.predictions_dir 
    
    _,sub_dir = get_checkpoint_dir(args)
    
    dirpred_paths = glob(os.path.join(predictions_dir, sub_dir, "*_DIRPRED.json"))
    
    angle_error, mag_error, epe, agl_error = [],[],[],[]
    
    items = []
    for dirpred_path in tqdm(dirpred_paths):
        magpred_path = dirpred_path.replace("_DIRPRED.json", "_MAGPRED.tif")
        aglpred_path = dirpred_path.replace("_DIRPRED.json", "_AGLPRED.tif")
        
        basename = os.path.basename(magpred_path)
        underscores = [ind for ind,val in enumerate(basename) if val=="_"]
        
        angle = np.int(basename.split("_")[-2])
        
        if not args.test_rotations and angle != 0:
            continue
        
        vflow_name = basename[:underscores[2]] + "_VFLOW" + basename[underscores[2]:underscores[3]] + ".json"
        agl_name = vflow_name.replace("_VFLOW_", "_AGL_").replace(".json", ".tif")
        vflow_gt_path = os.path.join(args.dataset_dir, "test", vflow_name)
        agl_gt_path = os.path.join(args.dataset_dir, "test", agl_name)
        
        items.append((vflow_gt_path,agl_gt_path,dirpred_path,magpred_path,aglpred_path,angle))
        
    if args.multiprocessing:
        pool = multiprocessing.Pool()
        results = list(tqdm(pool.imap_unordered(get_current_metrics, items), total=len(items)))
        pool.close()
        pool.join()
    else:
        results = [get_current_metrics(item) for item in tqdm(items)]
        
    for result in results:
        curr_angle_error,curr_mag_error,curr_epe,curr_agl_error = result
        angle_error.append(curr_angle_error)
        mag_error.append(curr_mag_error)
        epe.append(curr_epe)
        if curr_agl_error is not None:
            agl_error.append(curr_agl_error)
        
    mean_angle_error = np.nanmean(angle_error)
    mean_mag_error = np.nanmean(mag_error)
    mean_epe = np.nanmean(epe)
    if len(agl_error) > 0:
        mean_agl_error = np.nanmean(agl_error)

    fid = open(os.path.join(args.predictions_dir, "metrics_" + sub_dir + ".txt"), "w")
    fid.write("Angle error: %f" % mean_angle_error)
    fid.write("Mag error: %f" % mean_mag_error)
    fid.write("EPE: %f" % mean_epe)
    if len(agl_error) > 0:
        fid.write("AGL error: %f" % mean_agl_error)
    fid.close()

    print("Angle error: %f" % mean_angle_error)
    print("Mag error: %f" % mean_mag_error)
    print("EPE: %f" % mean_epe)
    if len(agl_error) > 0:
        print("AGL error: %f" % mean_agl_error)
        

def image_generator(data, args):
    idx = np.random.permutation(len(data))
    while True:
        batch_inds = get_batch_inds(idx, args.batch_size)
        for inds in batch_inds:
            img_batch,label_batch = load_batch(inds, data, args)
            yield (img_batch, label_batch)
    
def load_batch(inds, data, args):
    
    xydir_batch = np.zeros((len(inds), 2))
    mag_batch = np.zeros((len(inds), args.image_size[0], args.image_size[1], 1))
    image_batch = np.zeros((len(inds), args.image_size[0], args.image_size[1], 3))
    if args.add_height:
        agl_batch = np.zeros((len(inds), args.image_size[0], args.image_size[1], 1))

    for batch_ind,ind in enumerate(inds):

        rgb_path,vflow_path,agl_path = data[ind]

        image = load_image(rgb_path)
        agl = load_image(agl_path)
        vflow,mag,xdir,ydir,angle_orig = load_vflow(vflow_path, agl)

        if args.augmentation:
            image,mag,xdir,ydir,agl = augment(image, mag, xdir, ydir, agl=agl)

        xydir_batch[batch_ind,0] = xdir
        xydir_batch[batch_ind,1] = ydir
          
        image_batch[batch_ind,:,:,:] = image
        mag_batch[batch_ind,:,:,0] = mag
        if args.add_height:
            agl_batch[batch_ind,:,:,0] = agl
            
    mag_batch[np.isnan(mag_batch)] = ignore_value 
    gt_batch = {"xydir":xydir_batch, "mag":mag_batch}
    if args.add_height:
        agl_batch[np.isnan(agl_batch)] = ignore_value
        gt_batch["agl"] = agl_batch
        
    image_batch = image_preprocess(image_batch)
    
    return image_batch, gt_batch

def build_model(args, train_data, val_data):
        
    train_datagen = image_generator(train_data, args)
    val_datagen = image_generator(val_data, args)

    input_tensor = Input(shape=(args.image_size[0], args.image_size[1], 3))
    input_shape = (args.image_size[0], args.image_size[1], 3)
    
    model = UnetFlow(input_shape=input_shape, input_tensor=input_tensor, 
                        backbone_name=args.backbone, encoder_weights="imagenet", add_height=args.add_height)
    
    if args.continue_training_file is not None:
        model.load_weights(args.continue_training_file)

    loss = {"xydir":"mse", "mag":no_nan_mse}
    loss_weights = {"xydir": 1.0, "mag":1.0}
    if args.add_height:
        loss["agl"] = no_nan_mse
        loss_weights["agl"] = 1.0
        
    model.compile("Adam", loss=loss, loss_weights=loss_weights)
        
    return train_datagen, val_datagen, model

