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

import os
import cv2
import random
import numpy as np
        
def augment(image, mag, xdir, ydir, agl=None, rotate_prob=1, flip_prob=0.3):
        
    if random.uniform(0,1) < rotate_prob:
        rotate_angle = random.randint(0,359)
        xdir,ydir = rotate_xydir(xdir, ydir, rotate_angle)
        image,mag,agl = rotate_image(image, mag, agl, rotate_angle)

    if random.uniform(0,1) < flip_prob:
        image,mag,agl = flip(image, mag, agl, dim='x')
        xdir *= -1
    if random.uniform(0,1) < flip_prob:
        image,mag,agl = flip(image, mag, agl, dim='y')
        ydir *= -1

    return image, mag, xdir, ydir, agl
    
def flip(image, mag, agl, dim):
    if dim == 'x':
        image = image[:,::-1,:]
        mag = mag[:,::-1]
        if agl is not None:
            agl = agl[:,::-1]
    elif dim == 'y':
        image = image[::-1,:,:]
        mag = mag[::-1,:]
        if agl is not None:
            agl = agl[::-1,:]
    return image,mag,agl

def get_crop_region(image_rotated, image):
    excess_buffer = np.array(image_rotated.shape[:2])-np.array(image.shape[:2])
    r1, c1 = (excess_buffer / 2).astype(np.int)
    r2, c2 = np.array([r1,c1]) + image.shape[:2]
    return r1,c1,r2,c2

def rotate_xydir(xdir, ydir, rotate_angle):
    base_angle = np.degrees(np.arctan2(xdir,ydir))
    xdir = np.sin(np.radians(base_angle+rotate_angle))
    ydir = np.cos(np.radians(base_angle+rotate_angle))
    return xdir,ydir

def rotate_image(image, mag, agl, angle, image_only=False):
    
    if image_only:
        h,w = image.shape[:2]
    else:
        h,w = mag.shape[:2]
    rw,rh = (w/2, h/2)

    rot_mat = cv2.getRotationMatrix2D((rw,rh), angle, 1.0)

    cos,sin = np.abs(rot_mat[0, 0:2])
    wnew = int((h * sin) + (w * cos))
    hnew = int((h * cos) + (w * sin))

    rot_mat[0, 2] += np.int((wnew / 2) - rw)
    rot_mat[1, 2] += np.int((hnew / 2) - rh)

    image_rotated = None if image is None else cv2.warpAffine(image, rot_mat, (wnew, hnew), flags=cv2.INTER_LINEAR)

    if image_rotated is not None:
        r1,c1,r2,c2 = get_crop_region(image_rotated, image)
        image_rotated = image_rotated[r1:r2,c1:c2,:]
    
    if image_only:
        return image_rotated
    
    agl_rotated = None if agl is None else cv2.warpAffine(agl, rot_mat, (wnew, hnew), flags=cv2.INTER_LINEAR)
    mag_rotated = cv2.warpAffine(mag, rot_mat, (wnew, hnew), flags=cv2.INTER_LINEAR)
    
    if image_rotated is None:
        r1,c1,r2,c2 = get_crop_region(mag_rotated, mag)
    
    mag_rotated = mag_rotated[r1:r2,c1:c2]
    
    if agl_rotated is not None:
        agl_rotated = agl_rotated[r1:r2,c1:c2]

    return image_rotated,mag_rotated,agl_rotated

