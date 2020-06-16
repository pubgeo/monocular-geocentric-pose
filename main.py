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

import argparse
import os
from utilities.ml_utils import train,test,metrics

if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentation", action="store_true", help="train with rotations (modifies ground truth orientation)")
    parser.add_argument("--test-rotations", action="store_true", help="apply test-time rotations (more test instances)")
    parser.add_argument("--add-height", action="store_true", help="train with height regression")
    parser.add_argument("--train", action="store_true", help="train")
    parser.add_argument("--test", action="store_true", help="generate test predictions")
    parser.add_argument("--metrics", action="store_true", help="evaluate predictions against ground truth")
    parser.add_argument("--multiprocessing", action="store_true", help="use multiprocessing for metrics")
    parser.add_argument("--gpus", type=str, help="gpu indices (comma separated)", default='0')
    parser.add_argument("--num-epochs", type=int, help="gpu indices (comma separated)", default=100)
    parser.add_argument("--save-period", type=int, help="gpu indices (comma separated)", default=5)
    parser.add_argument("--batch-size", type=int, help="batch size", default=2)
    parser.add_argument("--continue-training-file", type=str, help="file to continue training from", default=None)
    parser.add_argument("--test-model-file", type=str, help="test checkpoint if not running default selection", default=None)
    parser.add_argument("--checkpoint-dir", type=str, help="where to store and load checkpoints from", default="./checkpoints")
    parser.add_argument("--tensorboard-dir", type=str, help="tensorboard log directory",  default="./tensorboard")
    parser.add_argument("--predictions-dir", type=str, help="where to store predictions", default="./predictions")
    parser.add_argument("--dataset-dir", type=str, help="dataset directory", default="./dataset")
    parser.add_argument("--train-sub-dir", type=str, help="train folder within datset-dir", default="train")
    parser.add_argument("--test-sub-dir", type=str, help="test folder within datset-dir", default="test")
    parser.add_argument("--image-size", type=int, nargs="+", help="image size", default=(2048,2048))
    parser.add_argument("--backbone", type=str, help="unet backbone", default="resnet34")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    if args.train:
        train(args)
        
    if args.test:
        test(args)
        
    if args.metrics:
        metrics(args)
    