# Learning Geocentric Object Pose in Oblique Monocular Images

## References

If you use our dataset or code, please cite [our paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Christie_Learning_Geocentric_Object_Pose_in_Oblique_Monocular_Images_CVPR_2020_paper.pdf):

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

The data used in our CVPR paper can be found here: https://ieee-dataport.org/open-access/urban-semantic-3d-dataset

The file names have the following format: ```{site name}_Tile_{tile #}_{data type}_{swath #}.{extension}```

Site names include JAX and OMA for the DFC19 data, and ATL for the ATL-SN6 data used in the paper. Tile numbers represent a center coordinate (i.e., common regions across multiple satellite swaths). Swath numbers represent unique satellite image swaths. Descriptions of each data type can be found in the dataset README files at the link above.

RGB, AGL, and VFLOW files are used for the training done in the paper, and CLS and SHDW files are used for test-time analysis.

## Dependencies

We used Anaconda Python for development and testing, and have included our YAML file. Our dependencies can also be installed as follows: ```conda install opencv gdal tensorflow-gpu keras tqdm scikit-image```

If you are not using Anaconda, you may have issues GDAL and need to follow other installation instructions. Alternatively, you can replace the GDAL i/o operations with something else. 

## Running the Code

The most important main.py arguments to change for the ablation studies performed in the paper are ```--add-height```, ```--augmentation```, and ```--test-rotations```.
+ ```--add-height``` will train a model that jointly learns to regress dense above-ground-level heights, where the estimates are then used to regress flow vector magnitudes in an end-to-end framework
+ ```--augmentation``` will perform train-time rotations by rotating the RGB and updating the orientation vector ground truth used to calculate dense flow vectors from magnitude
+ ```--test-rotations``` will perform rotations in increments of 36 degrees at test time in order to test the model's ability to generalize to unseen orientations

See ```main.py``` for descriptions of other arguments.

For our full approach (FLOW-HA in the paper), the following is used:

+ Training: ```python main.py --train --add-height --augmentation```
+ Testing (with test-time rotations): ```python main.py --test --add-height --augmentation --test-rotations```
+ Metrics: ```python main.py --metrics --add-height --augmentation --test-rotations --multiprocessing```

With the current ```main.py```, there is an assumption that all training data (RGB, AGL, VFLOW) is in ```./dataset/train```, and that all testing data is in ```./dataset/test```.

## License

The license is Apache 2.0. See LICENSE.

## Public Release

Approved for public release, 20-563