import os
import numpy as np
import cv2 # type: ignore
import torch # type: ignore

from torch.utils.data import Dataset # type: ignore
from glob import glob
from torchvision import transforms as tf # type: ignore
import warnings
warnings.filterwarnings("ignore")

class ThyroidNodules(Dataset):
    def __init__(self, images_path, masks_path, image_size, transform=None):
        """
        Initialize the Lung Nodules Dataset Loader.

        Args:
            images_path (list of str): List of paths to lung nodule images.
            masks_path (list of str): List of paths to lung nodule masks.
            image_size (tuple): Desired image size (height and width).
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.to_return_size = image_size
        self.transform = transform
        self.images = []
        self.masks = []
        for i in images_path:
            image = cv2.imread(i, cv2.IMREAD_COLOR)
            image = cv2.resize(image, self.to_return_size)
            self.images.append(image)
        for i in masks_path:
            mask = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.to_return_size)
            self.masks.append(mask)
        self.data = {'images': self.images, 'masks': self.masks}
        self.n_samples = len(self.images)

    def __getitem__(self, index):
        """Reading image"""
        image = self.images[index]
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)

        """Reading mask"""
        mask  = self.masks[index]
        mask = mask / mask.max()
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    dataset = ThyroidNodules(
        image_size=(256, 256),
        images_path=sorted(
            glob(
                os.path.join(
                    "/Users/eloise-em/Documents/Haris Ghafoor Archive/Research and Development/RnD/Thyroid Dataset/tn3k/trainval-image",
                    "*",
                )
            )
        ),
        masks_path=sorted(
            glob(
                os.path.join(
                    "/Users/eloise-em/Documents/Haris Ghafoor Archive/Research and Development/RnD/Thyroid Dataset/tn3k/trainval-mask",
                    "*",
                )
            )
        ),
        transform=tf.Compose(
            [
                tf.ToTensor(),  # Convert to tensor
                tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
               
            ]
        ),
    )
    # print(dataset[0].shape)
    print("Sample Image Shape", dataset.__getitem__(0)[0].shape)
    print("Sample Mask Shape", dataset.__getitem__(0)[1].shape)
