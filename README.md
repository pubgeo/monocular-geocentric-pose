# Learning Geocentric Object Pose in Oblique Monocular Images

## References

If you use our dataset or code, please cite our paper:

```
@inproceedings{christie2020geocentricpose,
  title={Learning Geocentric Object Pose in Oblique Monocular Images},
  author={Christie, Gordon and Munoz, Rai and Foster, Kevin H and Hagstrom, Shea T and Hager, Gregory D and Brown, Myron Z},
  booktitle={CVPR},
  year={2020}
}
```

Our code modifies the following repo in order to perform geocentric pose estimation: https://github.com/qubvel/segmentation_models We specifically added segmentation_models/unet_flow/builder.py

## Data

Coming very soon

## Dependencies

We used Anaconda Python for development and testing, and have included our YAML file. Our dependencies can also be installed as follows: ```conda install opencv gdal tensorflow-gpu keras tqdm scikit-image```

If you are not using Anaconda, you may have issues GDAL and need to follow other installation instructions. Alternatively, you can replace the GDAL i/o operations with something else. 

## Running the Code

The most important main.py arguments to change for ablation studies performed in the paper are ```--add-height```. ```--augmentation```, and ```--test-rotations```.

For our full approach, the following would be used:

+ Training: ```python main.py --train --add-height --augmentation```
+ Testing (with test-time rotations): ```python main.py --test --add-height --augmentation --test-rotations```
+ Metrics: ```python main.py --metrics --add-height --augmentation --test-rotations --multiprocessing```


## License

The license is Apache 2.0. See LICENSE.

## Public Release

Approved for public release, 20-563