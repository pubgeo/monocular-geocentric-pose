from pathlib import Path
import argparse
import functools
import os
import numpy as np
from tqdm import tqdm
from glob import glob
import random
import json
import multiprocessing
from osgeo import gdal
import cv2
from misc_utils import load_image, save_image, load_vflow


def rect_flow(rgb, mag, angle, agl):
    # filter magnitude of input flow vectors
    mag = cv2.medianBlur(mag, 5)
    # initialize output images with zeros
    output_rgb = np.zeros(rgb.shape, dtype=np.uint8)
    output_mag = np.zeros(mag.shape, dtype=np.float32)
    output_agl = np.zeros(agl.shape, dtype=np.float32)
    output_mask = np.ones(mag.shape, dtype=np.uint8)
    # get the flow vectors to map original features to new images
    y2 = mag * np.sin(angle)
    x2 = mag * np.cos(angle)
    x2 = (x2 + 0.5).astype(np.int32)
    y2 = (y2 + 0.5).astype(np.int32)
    rows, cols = np.mgrid[0 : mag.shape[0], 0 : mag.shape[1]]
    rows2 = np.clip(rows + x2, 0, mag.shape[0] - 1)
    cols2 = np.clip(cols + y2, 0, mag.shape[1] - 1)
    # map input pixel values to output images
    for i in range(0, mag.shape[0]):
        for j in range(0, mag.shape[0]):
            # favor taller things in output; this is a hard requirement
            if mag[rows[i, j], cols[i, j]] < output_mag[rows2[i, j], cols2[i, j]]:
                continue
            output_rgb[rows2[i, j], cols2[i, j], :] = rgb[rows[i, j], cols[i, j], :]
            output_agl[rows2[i, j], cols2[i, j]] = agl[rows[i, j], cols[i, j]]
            output_mag[rows2[i, j], cols2[i, j]] = mag[rows[i, j], cols[i, j]]
            output_mask[rows2[i, j], cols2[i, j]] = 0
    # filter AGL
    filtered_agl = cv2.medianBlur(output_agl, 5)
    filtered_agl = cv2.medianBlur(filtered_agl, 5)
    filtered_agl = cv2.medianBlur(filtered_agl, 5)
    filtered_agl = cv2.medianBlur(filtered_agl, 5)
    # filter occlusion mask to fill in pixels missed due to sampling error
    filtered_mask = cv2.medianBlur(output_mask, 5)
    filtered_mask = cv2.medianBlur(filtered_mask, 5)
    filtered_mask = cv2.medianBlur(filtered_mask, 5)
    filtered_mask = cv2.medianBlur(filtered_mask, 5)
    # replace non-occluded but also non-mapped RGB pixels with median of neighbors
    interp_mask = output_mask > filtered_mask
    filtered_rgb = cv2.medianBlur(output_rgb, 5)
    output_rgb[interp_mask, 0] = filtered_rgb[interp_mask, 0]
    output_rgb[interp_mask, 1] = filtered_rgb[interp_mask, 1]
    output_rgb[interp_mask, 2] = filtered_rgb[interp_mask, 2]
    return output_rgb, filtered_agl, filtered_mask


def write_rectified_images(rgb, mag, angle_rads, agl, output_rgb_path):
    # rectify all images
    rgb_rct, agl_rct, mask_rct = rect_flow(rgb, mag, angle_rads, agl)
    # format AGL image
    max_agl = np.nanpercentile(agl, 99)
    agl_rct[mask_rct > 0] = 0.0
    agl_rct[agl_rct > max_agl] = max_agl
    agl_rct *= 255.0 / max_agl
    agl_rct = 255.0 - agl_rct
    agl_rct = agl_rct.astype(np.uint8)
    # format RGB image
    rgb_rct[mask_rct > 0, 0] = 135
    rgb_rct[mask_rct > 0, 1] = 206
    rgb_rct[mask_rct > 0, 2] = 250
    save_image(output_rgb_path, rgb)
    save_image(output_rgb_path.replace(".tif", "_RECT.tif"), rgb_rct)
    # AGL with rectification
    rgb_rct[:, :, 0] = agl_rct
    rgb_rct[:, :, 1] = agl_rct
    rgb_rct[:, :, 2] = agl_rct
    rgb_rct[mask_rct > 0, 0] = 135
    rgb_rct[mask_rct > 0, 1] = 206
    rgb_rct[mask_rct > 0, 2] = 250
    save_image(
        output_rgb_path.replace("RGB", "AGL").replace(".tif", "_RECT.tif"), rgb_rct
    )
    # AGL without rectification
    agl_norect = np.copy(agl)
    agl_norect[agl_norect > max_agl] = max_agl
    agl_norect *= 255.0 / max_agl
    agl_norect = 255.0 - agl_norect
    agl_norect = agl_norect.astype(np.uint8)
    save_image(output_rgb_path.replace("RGB", "AGL"), agl_norect)


def get_current_metrics(item, args):
    # get arguments
    vflow_gt_path, agl_gt_path, vflow_pred_path, aglpred_path, rgb_path, args = item
    # load AGL, SCALE, and ANGLE predicted values
    agl_pred = load_image(aglpred_path, args)
    if agl_pred is None:
        return None
    vflow_items = load_vflow(
        vflow_pred_path, agl=agl_pred, args=args, return_vflow_pred_mat=True
    )
    if vflow_items is None:
        return None
    vflow_pred, mag_pred, xdir_pred, ydir_pred, vflow_data = vflow_items
    scale_pred, angle_pred = vflow_data["scale"], vflow_data["angle"]
    # load AGL, SCALE, and ANGLE ground truth values
    agl_gt = load_image(agl_gt_path, args)
    if agl_gt is None:
        return None
    vflow_gt_items = load_vflow(vflow_gt_path, agl_gt, args, return_vflow_pred_mat=True)
    if vflow_gt_items is None:
        return None
    vflow_gt, mag_gt, xdir_gt, ydir_gt, vflow_gt_data = vflow_gt_items
    scale_gt, angle_gt = vflow_gt_data["scale"], vflow_gt_data["angle"]
    # produce rectified images
    if args.rectify and args.output_dir is not None:
        rgb = load_image(rgb_path, args)
        output_rgb_path = os.path.join(args.output_dir, os.path.basename(rgb_path))
        write_rectified_images(rgb, mag_pred, angle_pred, agl_pred, output_rgb_path)
    # compute differences
    dir_pred = np.array([xdir_pred, ydir_pred])
    dir_pred /= np.linalg.norm(dir_pred)
    dir_gt = np.array([xdir_gt, ydir_gt])
    dir_gt /= np.linalg.norm(dir_gt)
    cos_ang = np.dot(dir_pred, dir_gt)
    sin_ang = np.linalg.norm(np.cross(dir_pred, dir_gt))
    rad_diff = np.arctan2(sin_ang, cos_ang)
    # get mean error values
    angle_error = np.degrees(rad_diff)
    scale_error = np.abs(scale_pred - scale_gt)
    mag_error = np.nanmean(np.abs(mag_pred - mag_gt))
    epe = np.nanmean(np.sqrt(np.sum(np.square(vflow_gt - vflow_pred), axis=2)))
    agl_error = np.nanmean(np.abs(agl_pred - agl_gt))
    # get RMS error values
    mag_rms = np.sqrt(np.nanmean(np.square(mag_pred - mag_gt)))
    epe_rms = np.sqrt(np.nanmean(np.sum(np.square(vflow_gt - vflow_pred), axis=2)))
    agl_rms = np.sqrt(np.nanmean(np.square(agl_pred - agl_gt)))
    # gather data for computing R-square for AGL
    agl_count = np.sum(np.isfinite(agl_gt))
    agl_sse = np.nansum(np.square(agl_pred - agl_gt))
    agl_gt_sum = np.nansum(agl_gt)
    # gather data for computing R-square for VFLOW
    vflow_count = np.sum(np.isfinite(vflow_gt))
    vflow_gt_sum = np.nansum(vflow_gt)
    vflow_sse = np.nansum(np.square(vflow_pred - vflow_gt))
    items = (
        angle_error,
        scale_error,
        mag_error,
        epe,
        agl_error,
        mag_rms,
        epe_rms,
        agl_rms,
        agl_count,
        agl_sse,
        agl_gt_sum,
        vflow_count,
        vflow_sse,
        vflow_gt_sum,
    )
    return items


def get_r2_denoms(item, agl_gt_mean, vflow_gt_mean):
    (
        vflow_gt_path,
        agl_gt_path,
        vflow_pred_path,
        aglpred_path,
        rgb_path,
        args,
    ) = item
    agl_gt = load_image(agl_gt_path, args)
    vflow_gt_items = load_vflow(vflow_gt_path, agl_gt, args, return_vflow_pred_mat=True)
    vflow_gt, mag_gt, xdir_gt, ydir_gt, vflow_gt_data = vflow_gt_items
    scale_gt, angle_gt = vflow_gt_data["scale"], vflow_gt_data["angle"]

    agl_denom = np.nansum(np.square(agl_gt - agl_gt_mean))
    vflow_denom = np.nansum(np.square(vflow_gt - vflow_gt_mean))

    items = (agl_denom, vflow_denom)
    return items


def get_city_scores(args, site):

    # build lists of images to process
    vflow_gt_paths = glob(os.path.join(args.truth_dir, site + "*_VFLOW*.json"))
    if vflow_gt_paths == []:
        return np.nan
    angle_error, scale_error, mag_error, epe, agl_error, mag_rms, epe_rms, agl_rms = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    items = []
    for vflow_gt_path in vflow_gt_paths:
        vflow_name = os.path.basename(vflow_gt_path)
        agl_name = vflow_name.replace("_VFLOW", "_AGL").replace(".json", ".tif")
        agl_gt_path = os.path.join(args.truth_dir, agl_name)
        rgb_path = agl_gt_path.replace("AGL", "RGB").replace(
            ".tif", f".{args.rgb_suffix}"
        )
        vflow_pred_path = os.path.join(args.predictions_dir, vflow_name)
        agl_pred_path = os.path.join(args.predictions_dir, agl_name)
        items.append(
            (vflow_gt_path, agl_gt_path, vflow_pred_path, agl_pred_path, rgb_path, args)
        )
    # compute metrics for each image
    pool = multiprocessing.Pool(args.num_processes)
    results = []
    for result in tqdm(
        pool.imap_unordered(functools.partial(get_current_metrics, args=args), items),
        total=len(items),
    ):
        results.append(result)
    pool.close()
    pool.join()

    # initialize AGL and VFLOW R-square data
    agl_count, agl_sse, agl_gt_sum, vflow_count, vflow_sse, vflow_gt_sum = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    # gather results from list
    angle_error, scale_error, mag_error, epe, agl_error, mag_rms, epe_rms, agl_rms = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for result in results:
        # get metrics for next image
        if result is None:
            return 0.0
        (
            curr_angle_error,
            curr_scale_error,
            curr_mag_error,
            curr_epe,
            curr_agl_error,
            curr_mag_rms,
            curr_epe_rms,
            curr_agl_rms,
            curr_agl_count,
            curr_agl_sse,
            curr_agl_gt_sum,
            curr_vflow_count,
            curr_vflow_sse,
            curr_vflow_gt_sum,
        ) = result
        # add metrics to lists
        angle_error.append(curr_angle_error)
        scale_error.append(curr_scale_error)
        mag_error.append(curr_mag_error)
        epe.append(curr_epe)
        mag_rms.append(curr_mag_rms)
        epe_rms.append(curr_epe_rms)
        agl_error.append(curr_agl_error)
        agl_rms.append(curr_agl_rms)
        # update data for AGL R-square
        agl_count = agl_count + curr_agl_count
        agl_sse = agl_sse + curr_agl_sse
        agl_gt_sum = agl_gt_sum + curr_agl_gt_sum
        # update data for VFLOW R-square
        vflow_count = vflow_count + curr_vflow_count
        vflow_sse = vflow_sse + curr_vflow_sse
        vflow_gt_sum = vflow_gt_sum + curr_vflow_gt_sum
    # compute statistics over all images
    mean_angle_error = np.nanmean(angle_error)
    rms_angle_error = np.sqrt(np.nanmean(np.square(angle_error)))
    mean_scale_error = np.nanmean(scale_error)
    rms_scale_error = np.sqrt(np.nanmean(np.square(scale_error)))
    mean_mag_error = np.nanmean(mag_error)
    rms_mag_error = np.sqrt(np.nanmean(np.square(mag_rms)))
    mean_epe = np.nanmean(epe)
    rms_epe = np.sqrt(np.nanmean(np.square(epe_rms)))
    mean_agl_error = np.nanmean(agl_error)
    rms_agl_error = np.sqrt(np.nanmean(np.square(agl_rms)))
    # compute AGL and EPE R-square
    print("Computing AGL and VFLOW R-squares...")
    agl_gt_mean = agl_gt_sum / (agl_count + 0.0001)
    vflow_gt_mean = vflow_gt_sum / (vflow_count + 0.0001)
    agl_denom = 0.0
    vflow_denom = 0.0
    scale_value = []

    # get agld and vflow denoms in parallel
    pool = multiprocessing.Pool(args.num_processes)
    agl_denoms = []
    vflow_denoms = []
    for (agl_denom, vflow_denom) in tqdm(
        pool.imap_unordered(
            functools.partial(
                get_r2_denoms, agl_gt_mean=agl_gt_mean, vflow_gt_mean=vflow_gt_mean
            ),
            items,
        ),
        total=len(items),
    ):
        agl_denoms.append(agl_denom)
        vflow_denoms.append(vflow_denom)
    pool.close()
    pool.join()

    agl_denom = np.sum(agl_denoms)
    vflow_denom = np.sum(vflow_denoms)
    agl_R2 = 1.0 - (agl_sse / (agl_denom + 0.0001))
    vflow_R2 = 1.0 - (vflow_sse / (vflow_denom + 0.0001))
    # write statistics to file
    if args.output_dir is not None:
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        fid = open(os.path.join(args.output_dir, "metrics.txt"), "a+")
        fid.write("\n\n")
        fid.write(site)
        fid.write("\nMEAN ERROR\n")
        fid.write("Angle error: %f\n" % mean_angle_error)
        fid.write("Scale error: %f\n" % mean_scale_error)
        fid.write("Mag error: %f\n" % mean_mag_error)
        fid.write("EPE : %f\n" % mean_epe)
        fid.write("AGL error: %f\n" % mean_agl_error)
        fid.write("ROOT MEAN SQUARE ERROR\n")
        fid.write("Angle error: %f\n" % rms_angle_error)
        fid.write("Scale error: %f\n" % rms_scale_error)
        fid.write("Mag error: %f\n" % rms_mag_error)
        fid.write("EPE error: %f\n" % rms_epe)
        fid.write("AGL error: %f\n" % rms_agl_error)
        fid.write("AGL R-square: %f\n" % agl_R2)
        fid.write("VFLOW R-square: %f\n" % vflow_R2)
        fid.close()
    # write statistics to screen
    print(f"\n{site} MEAN ERROR")
    print("Angle error: %f" % mean_angle_error)
    print("Scale error: %f" % mean_scale_error)
    print("Mag error: %f" % mean_mag_error)
    print("EPE: %f" % mean_epe)
    print("AGL error: %f" % mean_agl_error)
    print("ROOT MEAN SQUARE ERROR")
    print("Angle error: %f" % rms_angle_error)
    print("Scale error: %f" % rms_scale_error)
    print("Mag error: %f" % rms_mag_error)
    print("EPE error: %f" % rms_epe)
    print("AGL error: %f" % rms_agl_error)
    print("AGL R-square: ", agl_R2)
    print("VFLOW R-square: ", vflow_R2)
    return (vflow_R2 + agl_R2) / 2.0


def evaluate(args):
    # get scores for each city and average them
    score_arg = get_city_scores(args, "ARG")
    score_jax = get_city_scores(args, "JAX")
    score_oma = get_city_scores(args, "OMA")
    score_atl = get_city_scores(args, "ATL")
    final_score = (score_arg + score_jax + score_oma + score_atl) / 4.0
    print("ARG R-square: ", score_arg)
    print("JAX R-square: ", score_jax)
    print("OMA R-square: ", score_oma)
    print("ATL R-square: ", score_atl)
    print("FINAL R-square: ", final_score)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        fid = open(os.path.join(args.output_dir, "metrics.txt"), "a+")
        fid.write("\n\nFINAL R-square: %f\n" % final_score)
        fid.close()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rectify", action="store_true", help="write rectified images")
    parser.add_argument(
        "--output-dir", type=str, help="folder for output files", default=None
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        help="folder with predictions to be evaluated",
        default=None,
    )
    parser.add_argument(
        "--truth-dir", type=str, help="folder with truth values", default=None
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        help="number of processes for multiprocessing",
    )
    parser.add_argument(
        "--rgb-suffix", type=str, help="suffix for rgb files", default="j2k"
    )
    parser.add_argument(
        "--nan-placeholder", type=int, help="placeholder value for nans", default=65535
    )
    parser.add_argument(
        "--unit",
        type=str,
        help="unit (m or cm) of AGL predictions AND label files: both must be the same",
        default="cm",
    )
    args = parser.parse_args()
    evaluate(args)
