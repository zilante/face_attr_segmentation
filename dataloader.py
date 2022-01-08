import os

import torch

from torchvision.io import read_image
import torchvision.transforms as T

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import constants.data as constants


class ImageDataset(Dataset):
    """CelebAMask-HQ Dataset. Read images, masks, apply preprocessing transformations.
    
    Args:
        img_names: pandas DataFrame with image names
        img_dir: Path to images folder
        mask_dir: Path to segmentation masks folder
        transform: Image preprocessing
        mask_transform: Mask preprocessing 
    """
    def __init__(
        self,
        img_names,
        img_dir,
        mask_dir,
        transform=None,
        mask_transform=None
    ):
        self.img_names = img_names
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_names)

    def read_mask(self, path):
        # if there is no mask for class, then return black image without mask;
        # else return only first channel as the other two are the same
        return read_image(path)[0] if os.path.isfile(path) else\
               torch.zeros(constants.MASK_SIZE, constants.MASK_SIZE)

    def __getitem__(self, idx):
        img_name = self.img_names.iloc[idx]

        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        
        img_idx = img_name[:-len(".jpg")]
        # masks are evenly split into folders 0, 1, ..., MASK_DIR_COUNT - 1
        mask_dir_idx =\
         str(int(img_idx) // (constants.DATASET_SIZE // constants.MASK_DIR_COUNT))
        masks = torch.stack([
                 self.read_mask(
                    os.path.join(
                        self.mask_dir, mask_dir_idx,

                        # name of mask file looks like
                        # '<img_idx>_<class_name>.png'
                        img_idx + '_' + class_name + '.png'
                    )
                ) for class_name in constants.CLASSES
        ])
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            masks = self.mask_transform(masks)
        return image, masks


def get_image_preprocessing(preprocessing_fn):
    """
    Args:
        preprocessing_fn: preprocessing for pretrained encoder
    """
    def preprocess_image(image):
        new_img_size = constants.IMG_SIZE
        image = T.Resize([new_img_size, new_img_size])(image)

        # preprocessing for pretrained encoder requires
        # the channel dimension to be last
        image = torch.movedim(image, 0, 2)
        if preprocessing_fn:
            image = preprocessing_fn(image)

        return torch.movedim(image, 2, 0).float()

    return preprocess_image


def preprocess_mask(mask):
    return mask / 255


def get_dataloader(
    img_names, img_dir, mask_dir, batch_size, shuffle=True, preprocessing_fn=None
):
    data = ImageDataset(
        img_names, img_dir, mask_dir,
        transform=get_image_preprocessing(preprocessing_fn),
        mask_transform=preprocess_mask
    )

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
