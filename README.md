# Single View Geocentric Pose in the Wild (PyTorch)

## Branches

The master branch represents the code used in our CVPR EarthVision 2021 paper that will serve as the baseline solution for an upcoming public competition. We made improvements over our CVPR 2020 paper retained in the cvpr20 branch. Pre-trained models for both versions are provided.

## References

If you use our dataset or code, please cite [our CVPR EarthVision 2021 paper](https://arxiv.org/abs/2105.08229) and [our CVPR 2020 main conference paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Christie_Learning_Geocentric_Object_Pose_in_Oblique_Monocular_Images_CVPR_2020_paper.pdf):

```
@inproceedings{christie2021geocentricpose,
  title={Single View Geocentric Pose in the Wild},
  author={Christie, Gordon and Foster, Kevin H and Hagstrom, Shea and Hager, Gregory D and Brown, Myron Z},
  booktitle={CVPRW},
  year={2021}
}
```

```
@inproceedings{christie2020geocentricpose,
  title={Learning Geocentric Object Pose in Oblique Monocular Images},
  author={Christie, Gordon and Munoz, Rai and Foster, Kevin H and Hagstrom, Shea T and Hager, Gregory D and Brown, Myron Z},
  booktitle={CVPR},
  year={2020}
}
```

## Competition

We developed an upcoming competition in collaboration with [DrivenData](https://www.drivendata.org/). Developers from DrivenData contributed software improvements for reducing file sizes and run times for submission evaluation and documentation below for working with the competition data.

## Data

The original Urban Semantic 3D (US3D) data used in our CVPR 2020 paper is available on [DataPort](https://ieee-dataport.org/open-access/urban-semantic-3d-dataset).

Competition data associated with our CVPR EarthVision 2021 paper will be made available soon. The original data includes RGB images in TIFF format and above ground level (AGL) height images in floating point TIFF format with units of meters. To reduce file sizes, the competition RGB images are J2K, and the AGL images are integer TIFF format with units of centimeters.

## Dependencies

Our dependencies can also be installed as follows:

```conda install gdal cython opencv gdal tqdm scikit-image pytorch torchvision cudatoolkit -c pytorch``` (select an appropriate version of cudatoolkit)

```pip install segmentation-models-pytorch```

```cd utilities```

```python cythonize_invert_flow.py build_ext --inplace```

## Training

The following subsections describe the training process using the original and competition data. Note that we used a batch size of four images when training with a GeForce RTX 2080 Ti with 11 GB VRAM. 

### Training with original data

The training process consists of two stages. 

1. Downsampling
2. Training

First, the data is downsampled by a factor of two for training our model. We note that **the downsampling save process always saves in units of meters**. This means that the second stage, actually training the model on the downsampled images, will be in units of `m` for both the original and competition data. For the original data, the first stage downsampling arguments specify that the input AGL and VFLOW data are in units of meter, and the RGB data is a `.tif` filetype. The second stage command looks generally the same as its original counterpart.

The biggest differences in the original data is the units of `m` (vs `cm` in the competition data) and the RGB filetype of `tif` (vs `j2k` in the competition data). The `--unit` and `--rgb-suffix` arguments allow the user to specify these contexts.

In this case, the `--indir` argument specifies the path to the unconverted, uncompressed training data directory and may need to be changed based on your data paths. As in the original version of this training process, all required RGB, AGL, and VFLOW files are assumed to live in the same directory.

#### Example commands

```
# Downsample: note unit m and rgb-suffix tif
python utilities/downsample_images.py \
    --indir="../data/raw/train" \
    --outdir="../data/interim/orig-train-half-res" \
    --unit="m" \
    --rgb-suffix="tif"

# Train: note unit m and rgb-suffix tif
python main.py \
    --num-epochs=200 \
    --train \
    --checkpoint-dir="../data/interim/original_data_training_assets/checkpoints" \ 
    --dataset-dir="../data/interim/orig-train-half-res" \
    --batch-size=4 \
    --gpus="0" \
    --augmentation \
    --num-workers=4 \
    --save-best \
    --train-sub-dir="" \
    --unit="m" \
    --rgb-suffix="tif"
```

### Training with competition data

Training with the competition data means the all input files --AGL, VFLOW, and RGB-- are in units of `cm` rather than `m`, and RGB files are `.j2k` extensions, rather than `.tif`. The **downsampling procdedure converts and saves in units to `m`**. So the training command is still essentially the same as above. 

```
# Downsample: note unit cm and j2k
python utilities/downsample_images.py \
    --indir="../data/processed/final/public/train" \
    --outdir="../data/processed/final/public/comp-train-half-res" \
    --unit="cm" \
    --rgb-suffix="j2k"

# Train: note unit m (since downsample converts to m) and rgb-suffix tif
python main.py \
    --num-epochs=200 \
    --train \
    --checkpoint-dir="../data/interim/competition_data_training_assets/checkpoints" \ 
    --dataset-dir="../data/processed/final/public/comp-train-half-res" \
    --batch-size=4 \
    --gpus="0" \
    --augmentation \
    --num-workers=4 \
    --save-best \
    --train-sub-dir="" \
    --rgb-suffix="tif" \
    --unit="m"
```

## Evaluation

The evaluation process consists of two stages

1. Generating predictions
2. Running the evaluation script

Currently, both the predicition and ground truth files MUST be in the same units on load, and units are converted to to meters before evaluation.

The model training process is configured to learn regression in meters. Even if the input files were originally in `cm`, they will be converted and learning will occur in `m`. However, we have added an argument, `--convert-predictions-to-cm-and-compress`, which converts to `cm` and compresses the data as specified in the function `utilities.misc_utils.convert_and_compress_prediction_dir`. The defaults are consistent with the requirements for submission to the DrivenData platform (except that the function does not perform the final submission step of creating a `tar.gz` of the dir).

### Evaluation with original data

Here we do not need to run `utilities.misc_utils.convert_and_compress_prediction_dir`, so set `--convert-predictions-to-cm-and-compress` to `False`.

```
# The dataset specified consists of the untouched, original test RGB
python main.py \
    --test \
    --model-path="../data/interim/original_data_training_assets/checkpoints/model_best.pth" \
    --predictions-dir="../data/processed/original_preds" \
    --dataset-dir="../data/raw/test_rgb" \
    --batch-size=8 \
    --gpus=0 \
    --downsample=2 \
    --test-sub-dir="" \
    --rgb-suffix="tif" \
    --convert-predictions-to-cm-and-compress=False

# Currently, both the predicition and ground truth files MUST be in the same units on load.
# Units are converted to to meters before evaluation.  
# Here we specify that the input unit for both the predictions and ground truth is cm.
# The dataset specified consists of the untouched, original test labels
python utilities/evaluate.py \
    --output-dir="../data/processed/original_pred_eval" \
    --predictions-dir="../data/processed/original_preds" \
    --truth-dir="../data/raw/test_ref" \
    --unit="m"

```

### Evaluation with competition data

Here we do need to run `utilities.misc_utils.convert_and_compress_prediction_dir`, so set `--convert-predictions-to-cm-and-compress` to `True`. Rather than overwriting the uncompressed/converted prediction directory, or converting on the fly, currently `utilities.misc_utils.convert_and_compress_prediction_dir` creates a new directory with suffixes describing the conversion and compression.

So if we specify `--predictions-dir="../data/processed/competition_preds"`, the unconverted/compressed preds will be generated and live in `competition_preds`. But a compressed/converted dir `competition_preds_converted_cm_compressed_uint16` will be generated *along side* `competition_preds`. 

To evaluate against the production test data, we use the converted directory, `competition_preds_converted_cm_compressed_uint16`.

```
python main.py \
    --test \
    --model-path="../data/interim/competition_data_training_assets/checkpoints/model_best.pth" \
    --predictions-dir="../data/processed/competition_preds" \
    --dataset-dir="../data/processed/final/public/test_rgbs" \
    --batch-size=8 \
    --gpus=0 \
    --downsample=2 \
    --test-sub-dir="" \
    --convert-predictions-to-cm-and-compress=True

# Currently, both the predicition and ground truth files MUST be in the same units on load.
# Units are converted to to meters before evaluation.  
# Here we specify that the input unit for both the predictions and ground truth is cm.
# The truth-dir here is the private, converted and compressed version of the labels loaded during evaluation on our production platform.
python utilities/evaluate.py \
    --output-dir="../data/processed/competition_pred_eval" \
    --predictions-dir="../data/processed/competition_preds_converted_cm_compressed_uint16" \
    --truth-dir="../data/processed/final/private/test_agls" \
    --unit="cm"
```

## License

The license is Apache 2.0. See LICENSE.

## Public Release

Approved for public release, 20-563
