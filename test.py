import segmentation_models_pytorch as smp

from typing import Optional
import argparse
import os
from pathlib import Path

import pandas as pd

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

import constants.data as data_constants
import constants.model as model_constants
from dataloader import get_dataloader


def get_loader(data_dir, img_names):
    img_dir = os.path.join(data_dir, data_constants.IMG_DIR)
    mask_dir = os.path.join(data_dir, data_constants.MASK_DIR)

    # get preprocessing for pretrained encoder
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        model_constants.ENCODER, model_constants.ENCODER_WEIGHTS
    )

    batch_size = 10
    loader = get_dataloader(
        img_names, img_dir, mask_dir, batch_size, shuffle=True,
        preprocessing_fn=preprocessing_fn
    )

    return loader


class Test:
    """Class for model testing. Get test score and visualization.

    Args:
        model_path: Path to the model weights
        data_dir: Path to the data folder
        split_info_file: Path to the file with train/test split information
        visualize_dir: Path to the folder to save visualizations
                       (if the folder doesn't exist, then it will be created)
        device: Device, on which the test is performed. One of **"cpu"** and **"cuda"**
    """
    def __init__(
        self,
        model_path: str,
        data_dir: str,
        split_info_file: str,
        visualize_dir: Optional[str] = None,
        device: Optional[str] = 'cpu'
    ):
        self.model = torch.load(model_path, map_location=device)

        # get test image names
        img_names = pd.read_csv(split_info_file)
        self.img_names = img_names[~img_names['is_train']]['name']
        self.data_dir = data_dir

        self.loader = get_loader(data_dir, self.img_names)
        self.device = device
        self.visualize_dir = visualize_dir

    def test(self):
        loss = smp.losses.dice.DiceLoss(
            eps=1e-7, mode='multilabel', log_loss=True, from_logits=False
        )
        # give the name for test logs
        loss.__name__ = "log_dice_loss"

        channels = set(range(len(data_constants.CLASSES)))
        metrics = [
            # compute Dice-score for each class
            # threshold=0.5 is used for outputs binarization
            smp.utils.metrics.Fscore(
                name = class_name + '_dice_score', threshold=0.5,
                ignore_channels = channels - {i}
            ) for i, class_name in enumerate(data_constants.CLASSES)
        ]

        epoch = smp.utils.train.ValidEpoch(
            self.model, 
            loss=loss, 
            metrics=metrics, 
            device=self.device,
            verbose=True,
        )
        epoch.run(self.loader)

    def visualize(self):
        imgs, _ = next(iter(self.loader))
        img = imgs[0].to(self.device).unsqueeze(0)

        # predict probability masks and get binarized masks
        pr_masks = self.model.predict(img)
        masks = (pr_masks.squeeze().cpu().round())

        # create directory, if it doesn't exist yet
        Path(self.visualize_dir).mkdir(exist_ok=True)
        save_image(imgs[0], os.path.join(self.visualize_dir, "img.png"))

        for i, (class_name, mask) in enumerate(
            zip(data_constants.CLASSES, masks)
        ):
            save_image(mask, os.path.join(self.visualize_dir, class_name + ".png"))


def parse_args():
    parser = argparse.ArgumentParser(description='Test the model.')

    parser.add_argument('model_path', metavar='M', type=str,
                        help='path to the model weights')

    parser.add_argument('data_dir', metavar='D', type=str,
                        help='path to the dir, containing the \'/CelebAMask-HQ\' dir')

    parser.add_argument('split_info_path', metavar='S', type=str,
                        help='path to the split information')

    parser.add_argument('-d', dest='device', action='store',
                        default='cpu', choices=['cpu', 'cuda'],
                        help='device, on which test is performed')

    parser.add_argument('-v', dest='visualize_dir', action='store',
                        default='', help='directory to save visualization')

    return parser.parse_args()


def main():
    args = parse_args()

    test = Test(
        args.model_path, args.data_dir, args.split_info_path,
        args.visualize_dir, args.device
    )

    test.test()
    if args.visualize_dir:
        test.visualize()
    

if __name__ == "__main__":
   main()
