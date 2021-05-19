import os
import numpy as np
import sys

from glob import glob
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

from tqdm import tqdm

import json
import cv2

from segmentation_models_pytorch.utils.meter import AverageValueMeter

from pathlib import Path
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from utilities.misc_utils import (
    UNITS_PER_METER_CONVERSION_FACTORS,
    convert_and_compress_prediction_dir,
    load_image,
    load_vflow,
    get_rms,
    get_r2_info,
    get_angle_error,
    get_r2,
    save_image,
)
from utilities.augmentation_vflow import augment_vflow
from utilities.unet_vflow import UnetVFLOW


RNG = np.random.RandomState(4321)


class Dataset(BaseDataset):
    def __init__(
        self,
        sub_dir,
        args,
        rng=RNG,
    ):

        self.is_test = sub_dir == args.test_sub_dir
        self.rng = rng

        # create all paths with respect to RGB path ordering to maintain alignment of samples
        dataset_dir = Path(args.dataset_dir) / sub_dir
        rgb_paths = list(dataset_dir.glob(f"*_RGB.{args.rgb_suffix}"))
        if rgb_paths == []: rgb_paths = list(dataset_dir.glob(f"*_RGB*.{args.rgb_suffix}")) # original file names
        agl_paths = list(
            pth.with_name(pth.name.replace("_RGB", "_AGL")).with_suffix(".tif")
            for pth in rgb_paths
        )
        vflow_paths = list(
            pth.with_name(pth.name.replace("_RGB", "_VFLOW")).with_suffix(".json")
            for pth in rgb_paths
        )

        if self.is_test:
            self.paths_list = rgb_paths
        else:
            self.paths_list = [
                (rgb_paths[i], vflow_paths[i], agl_paths[i])
                for i in range(len(rgb_paths))
            ]

            self.paths_list = [
                self.paths_list[ind]
                for ind in self.rng.permutation(len(self.paths_list))
            ]
            if args.sample_size is not None:
                self.paths_list = self.paths_list[: args.sample_size]
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            args.backbone, "imagenet"
        )

        self.args = args
        self.sub_dir = sub_dir

    def __getitem__(self, i):

        if self.is_test:
            rgb_path = self.paths_list[i]
            image = load_image(rgb_path, self.args)
        else:
            rgb_path, vflow_path, agl_path = self.paths_list[i]
            image = load_image(rgb_path, self.args)
            agl = load_image(agl_path, self.args)
            mag, xdir, ydir, vflow_data = load_vflow(vflow_path, agl, self.args)
            scale = vflow_data["scale"]
            if self.args.augmentation:
                image, mag, xdir, ydir, agl, scale = augment_vflow(
                    image,
                    mag,
                    xdir,
                    ydir,
                    vflow_data["angle"],
                    vflow_data["scale"],
                    agl=agl,
                )
            xdir = np.float32(xdir)
            ydir = np.float32(ydir)
            mag = mag.astype("float32")
            agl = agl.astype("float32")
            scale = np.float32(scale)

            xydir = np.array([xdir, ydir])

        if self.is_test and self.args.downsample > 1:
            image = cv2.resize(
                image,
                (
                    int(image.shape[0] / self.args.downsample),
                    int(image.shape[1] / self.args.downsample),
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        image = self.preprocessing_fn(image).astype("float32")
        image = np.transpose(image, (2, 0, 1))

        if self.is_test:
            return image, str(rgb_path)
        else:
            return image, xydir, agl, mag, scale

    def __len__(self):
        return len(self.paths_list)


class Epoch:
    def __init__(
        self,
        model,
        args,
        dense_loss=None,
        angle_loss=None,
        scale_loss=None,
        stage_name=None,
        device="cpu",
        verbose=True,
    ):
        self.args = args
        self.model = model
        self.dense_loss = dense_loss
        self.angle_loss = angle_loss
        self.scale_loss = scale_loss
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self.loss_names = ["combined", "agl", "mag", "angle", "scale"]

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        if self.stage_name != "valid":
            self.dense_loss.to(self.device)
            self.angle_loss.to(self.device)
            self.scale_loss.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}

        loss_meters = {}
        for loss_name in self.loss_names:
            loss_meters[loss_name] = AverageValueMeter()

        if self.stage_name == "valid":
            agl_count, agl_error_sum, agl_gt_sq_sum, agl_sum = 0, 0, 0, 0
            mag_count, mag_error_sum, mag_gt_sq_sum, mag_sum = 0, 0, 0, 0
            angle_errors = []
            agl_rms = []
            mag_rms = []
            scale_errors = []

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for itr_data in iterator:
                image, xydir, agl, mag, scale = itr_data
                scale = torch.unsqueeze(scale, 1)

                image = image.to(self.device)

                if self.stage_name != "valid":
                    xydir, agl, mag, scale = (
                        xydir.to(self.device),
                        agl.to(self.device),
                        mag.to(self.device),
                        scale.to(self.device),
                    )
                    y = [xydir, agl, mag, scale]

                    (
                        loss,
                        xydir_pred,
                        agl_pred,
                        mag_pred,
                        scale_pred,
                    ) = self.batch_update(image, y)

                    loss_logs = {}

                    for name in self.loss_names:
                        curr_loss = loss[name].cpu().detach().numpy()
                        if name == "scale":
                            curr_loss = np.mean(curr_loss)
                        loss_meters[name].add(curr_loss)
                        loss_logs[name] = loss_meters[name].mean

                    logs.update(loss_logs)
                else:

                    xydir_pred, agl_pred, mag_pred, scale_pred = self.batch_update(
                        image
                    )

                    xydir = xydir.cpu().detach().numpy()
                    agl = agl.cpu().detach().numpy()
                    mag = mag.cpu().detach().numpy()
                    scale = scale.cpu().detach().numpy()

                    xydir_pred = xydir_pred.cpu().detach().numpy()
                    agl_pred = agl_pred.cpu().detach().numpy()
                    mag_pred = mag_pred.cpu().detach().numpy()
                    scale_pred = scale_pred.cpu().detach().numpy()

                    for batch_ind in range(agl.shape[0]):

                        count, error_sum, rms, data_sum, gt_sq_sum = get_r2_info(
                            agl[batch_ind, :, :], agl_pred[batch_ind, :, :]
                        )
                        agl_count += count
                        agl_error_sum += error_sum
                        agl_rms.append(rms)
                        agl_sum += data_sum
                        agl_gt_sq_sum += gt_sq_sum

                        count, error_sum, rms, data_sum, gt_sq_sum = get_r2_info(
                            mag[batch_ind, :, :], mag_pred[batch_ind, :, :]
                        )
                        mag_count += count
                        mag_error_sum += error_sum
                        mag_rms.append(rms)
                        mag_sum += data_sum
                        mag_gt_sq_sum += gt_sq_sum

                        dir_pred = xydir_pred[batch_ind, :]
                        dir_gt = xydir[batch_ind, :]

                        angle_error = get_angle_error(dir_pred, dir_gt)

                        angle_errors.append(angle_error)
                        scale_errors.append(
                            np.abs(scale[batch_ind] - scale_pred[batch_ind])
                        )

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        if self.stage_name == "valid":

            r2_agl = get_r2(agl_error_sum, agl_gt_sq_sum, agl_sum, agl_count)
            r2_mag = get_r2(mag_error_sum, mag_gt_sq_sum, mag_sum, mag_count)

            angle_rms = get_rms(angle_errors)
            scale_rms = get_rms(scale_errors)
            agl_rms = get_rms(agl_rms)
            mag_rms = get_rms(mag_rms)

            print(
                "VAL Angle RMS: %.2f; AGL RMS: %.2f, R^2: %.4f; MAG RMS: %.2f, R^2: %.4f; Scale RMS: %.4f"
                % (angle_rms, agl_rms, r2_agl, mag_rms, r2_mag, scale_rms)
            )

        return logs


class TrainEpoch(Epoch):
    def __init__(
        self,
        model,
        args,
        dense_loss,
        angle_loss,
        scale_loss,
        optimizer,
        device="cpu",
        verbose=True,
    ):
        super().__init__(
            model=model,
            args=args,
            dense_loss=dense_loss,
            angle_loss=angle_loss,
            scale_loss=scale_loss,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        xydir_pred, agl_pred, mag_pred, scale_pred = self.model.forward(x.float())

        scale_pred = torch.unsqueeze(scale_pred, 1)

        xydir, agl, mag, scale = y
        loss_agl = self.dense_loss(agl_pred, agl)
        loss_mag = self.dense_loss(mag_pred, mag)
        loss_angle = self.angle_loss(xydir_pred, xydir)

        loss_scale = self.scale_loss(scale_pred, scale)

        loss_combined = (
            self.args.agl_weight * loss_agl
            + self.args.mag_weight * loss_mag
            + self.args.angle_weight * loss_angle
            + self.args.scale_weight * loss_scale
        )

        loss = {
            "combined": loss_combined,
            "agl": loss_agl,
            "mag": loss_mag,
            "angle": loss_angle,
            "scale": loss_scale,
        }

        loss_combined.backward()
        self.optimizer.step()

        return loss, xydir_pred, agl_pred, mag_pred, scale_pred


class ValidEpoch(Epoch):
    def __init__(self, model, args, device="cpu", verbose=True):
        super().__init__(
            model=model,
            args=args,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x):
        with torch.no_grad():
            xydir_pred, agl_pred, mag_pred, scale_pred = self.model.forward(x.float())

            scale_pred = torch.unsqueeze(scale_pred, 1)

        return xydir_pred, agl_pred, mag_pred, scale_pred


class NoNaNMSE(smp.utils.base.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def forward(self, output, target):
        diff = torch.squeeze(output) - target
        not_nan = ~torch.isnan(diff)
        loss = torch.mean(diff.masked_select(not_nan) ** 2)
        return loss


class MSELoss(smp.utils.base.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, output, target):
        return torch.nn.MSELoss()(output, target)


def train(args):

    torch.backends.cudnn.benchmark = True

    model = build_model(args)

    train_dataset = Dataset(sub_dir=args.train_sub_dir, args=args)
    val_dataset = Dataset(sub_dir=args.valid_sub_dir, args=args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=args.learning_rate),
        ]
    )

    dense_loss = NoNaNMSE()
    angle_loss = MSELoss()
    scale_loss = MSELoss()

    train_epoch = TrainEpoch(
        model,
        args=args,
        dense_loss=dense_loss,
        angle_loss=angle_loss,
        scale_loss=scale_loss,
        optimizer=optimizer,
        device="cuda",
    )

    val_epoch = ValidEpoch(
        model,
        args=args,
        device="cuda",
    )

    for i in range(args.num_epochs):

        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)

        if args.val_period > 0 and ((i + 1) % args.val_period) == 0:
            valid_logs = val_epoch.run(val_loader)

        if ((i + 1) % args.save_period) == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.checkpoint_dir, "./model_%d.pth" % i),
            )

        # save best epoch
        if args.save_best:
            combined_loss = train_logs["combined"]
            if i == 0:
                best_loss = combined_loss
            if combined_loss <= best_loss:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.checkpoint_dir, "./model_best.pth"),
                )


def test(args):

    torch.backends.cudnn.benchmark = True

    if args.model_path is None:
        model_paths = glob(os.path.join(args.checkpoint_dir, "*.pth"))
        nums = [int(path.split("_")[-1].replace(".pth", "")) for path in model_paths]
        idx = np.argsort(nums)[::-1]
        model_path = model_paths[idx[0]]
    else:
        model_path = args.model_path

    model = build_model(args)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.to("cuda")
    model.eval()
    with torch.no_grad():

        test_dataset = Dataset(sub_dir=args.test_sub_dir, args=args)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
        )
        predictions_dir = Path(args.predictions_dir)
        for images, rgb_paths in tqdm(test_loader):

            images = images.float().to("cuda")
            pred = model(images)

            numpy_preds = []
            for i in range(len(pred)):
                numpy_preds.append(pred[i].detach().cpu().numpy())

            xydir_pred, agl_pred, mag_pred, scale_pred = numpy_preds

            if scale_pred.ndim == 0:
                scale_pred = np.expand_dims(scale_pred, axis=0)

            for batch_ind in range(agl_pred.shape[0]):
                # vflow pred
                angle = np.arctan2(xydir_pred[batch_ind][0], xydir_pred[batch_ind][1])
                vflow_data = {
                    "scale": np.float64(
                        scale_pred[batch_ind] * args.downsample
                    ),  # upsample
                    "angle": np.float64(angle),
                }

                # agl pred
                curr_agl_pred = agl_pred[batch_ind, 0, :, :]
                curr_agl_pred[curr_agl_pred < 0] = 0
                agl_resized = cv2.resize(
                    curr_agl_pred,
                    (
                        curr_agl_pred.shape[0] * args.downsample,  # upsample
                        curr_agl_pred.shape[1] * args.downsample,  # upsample
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

                # save
                rgb_path = predictions_dir / Path(rgb_paths[batch_ind]).name
                agl_path = rgb_path.with_name(
                    rgb_path.name.replace("_RGB", "_AGL")
                ).with_suffix(".tif")
                vflow_path = rgb_path.with_name(
                    rgb_path.name.replace("_RGB", "_VFLOW")
                ).with_suffix(".json")

                json.dump(vflow_data, vflow_path.open("w"))
                save_image(agl_path, agl_resized)  # save_image assumes units of meters

    # creates new dir predictions_dir_con
    if args.convert_predictions_to_cm_and_compress:
        convert_and_compress_prediction_dir(predictions_dir=predictions_dir)


def build_model(args):
    model = UnetVFLOW(args.backbone, encoder_weights="imagenet")
    return model
