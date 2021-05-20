import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from misc_utils import load_image, load_vflow, save_image


def downsample_images(args, downsample=2):
    indir = Path(args.indir)
    outdir = Path(args.outdir)

    outdir.mkdir(exist_ok=True)
    rgb_paths = list(indir.glob(f"*_RGB.{args.rgb_suffix}"))
    if rgb_paths == []: rgb_paths = list(indir.glob(f"*_RGB*.{args.rgb_suffix}")) # original file names

    for rgb_path in tqdm(rgb_paths):

        # load
        agl_path = rgb_path.with_name(
            rgb_path.name.replace("_RGB", "_AGL")
        ).with_suffix(".tif")
        vflow_path = rgb_path.with_name(
            rgb_path.name.replace("_RGB", "_VFLOW")
        ).with_suffix(".json")
        rgb = load_image(rgb_path, args)  # args.unit used to convert units on load
        agl = load_image(agl_path, args)  # args.unit used to convert units on load
        _, _, _, vflow_data = load_vflow(
            vflow_path, agl, args
        )  # arg.unit used to convert units on load

        # downsample
        target_shape = (int(rgb.shape[0] / downsample), int(rgb.shape[1] / downsample))
        rgb = cv2.resize(rgb, target_shape)
        agl = cv2.resize(agl, target_shape, interpolation=cv2.INTER_NEAREST)
        vflow_data["scale"] /= downsample

        # save
        # units are NOT converted back here, so are in m
#        save_image((outdir / rgb_path.name), rgb)
        save_image((outdir / rgb_path.name.replace("j2k","tif")), rgb) # save as tif to be consistent with old code

        save_image((outdir / agl_path.name), agl)
        with open((outdir / vflow_path.name), "w") as outfile:
            json.dump(vflow_data, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, help="input directory", default=None)
    parser.add_argument("--outdir", type=str, help="output directory", default=None)
    parser.add_argument(
        "--nan-placeholder", type=int, help="placeholder value for nans", default=65535
    )
    parser.add_argument(
        "--unit", type=str, help="unit of AGLS (m, cm, or dm)", default="cm"
    )
    parser.add_argument(
        "--rgb-suffix",
        type=str,
        help="file extension for RGB data, e.g., tif or j2k",
        default="j2k",
    )
    args = parser.parse_args()
    downsample_images(args)
