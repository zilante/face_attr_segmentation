import segmentation_models_pytorch as smp

import os
import argparse

import pandas as pd

import torch

import constants.data as data_constants
import constants.model as model_constants
from dataloader import get_dataloader


def get_train_val_loaders(data_dir, split_info_file):
    # get train image names
    img_names = pd.read_csv(split_info_file)
    img_names = img_names[img_names['is_train']]['name']

    # split images into train and validation
    train_size = int(img_names.size * 0.9)
    train_img_names, valid_img_names = img_names[:train_size], img_names[train_size:]

    img_dir = os.path.join(data_dir, data_constants.IMG_DIR)
    mask_dir = os.path.join(data_dir, data_constants.MASK_DIR)

    # get preprocessing for pretrained encoder
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        model_constants.ENCODER, model_constants.ENCODER_WEIGHTS
    )

    batch_size = 15
    train_loader = get_dataloader(
        train_img_names, img_dir, mask_dir, batch_size, shuffle=True,
        preprocessing_fn=preprocessing_fn
    )

    batch_size = 15
    valid_loader = get_dataloader(
        valid_img_names, img_dir, mask_dir, batch_size, shuffle=False,
        preprocessing_fn=preprocessing_fn
    )

    return train_loader, valid_loader


def train_loop(epoch_count, train_epoch, valid_epoch, train_loader, valid_loader):
    max_score = 0
    model = train_epoch.model

    for i in range(0, epoch_count):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
 
        if max_score < valid_logs['fscore']:
            max_score = valid_logs['fscore']
            torch.save(model, './best_model.pth')
            print('Model saved!')


def train(data_dir, split_info_file, device='cpu', verbose=False):
    train_loader, valid_loader = get_train_val_loaders(data_dir, split_info_file)

    model = smp.FPN(
        encoder_name=model_constants.ENCODER, 
        encoder_weights=model_constants.ENCODER_WEIGHTS,
        in_channels=3,
        classes=len(data_constants.CLASSES),
        activation='sigmoid'
    )

    loss = smp.losses.dice.DiceLoss(
        eps=1e-7, mode='multilabel', from_logits=False
    )
    # give the name for train logs
    loss.__name__ = "dice_loss"

    channels = set(range(len(data_constants.CLASSES)))
    metrics = [
        # compute Dice-score for each class
        # threshold=0.5 is used for outputs binarization
        smp.utils.metrics.Fscore(
            name = class_name + ' dice score', threshold=0.5,
            ignore_channels = channels - {i}
        ) for i, class_name in enumerate(data_constants.CLASSES)
    # Dice-score mean over all classes is also computed
    ] + [smp.utils.metrics.Fscore(threshold=0.5)]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=verbose,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=verbose,
    )

    epoch_count = 8
    train_loop(epoch_count, train_epoch, valid_epoch, train_loader, valid_loader)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the model.')

    parser.add_argument('data_path', metavar='D', type=str,
                        help='path to the data')

    parser.add_argument('split_info_path', metavar='S', type=str,
                        help='path to the split information')

    parser.add_argument('-d', dest='device', action='store',
                        default='cpu', choices=['cpu', 'cuda'],
                        help='device, on which train is performed')

    parser.add_argument('-v', dest='verbose', action='store_true',
                        help='verbose flag')

    return parser.parse_args()


def main():
    args = parse_args()
    train(
          args.data_path, args.split_info_path, device=args.device,
          verbose=args.verbose
    )


if __name__ == "__main__":
   main()
