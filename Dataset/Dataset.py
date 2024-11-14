# --------------------------------------------------------------------------------------------------
# ------------------------------------ MODULE :: Dataset Manager -----------------------------------
# --------------------------------------------------------------------------------------------------
import torchvision.transforms as transforms
import torch.utils.data as datatools

import pathlib
import typing
import torch
import random

from PIL import Image

T_Tensors = tuple[torch.Tensor, torch.Tensor]
T_Paths = tuple[pathlib.Path, pathlib.Path]


# --------------------------------------------------------------------------------------------------
# ------------------------------------ CLASS :: Dataset Manager ------------------------------------
# --------------------------------------------------------------------------------------------------
class Dataset(datatools.Dataset, datatools.Sampler):
    """A combination sampler and dataset (image finder and imager loader) which simultaneously...
        1. Yields batches of images all belonging to the class (style)
        2. Loads, transforms, and tensorizes images
    """

    # ------------------------------------------------------------------------------------------
    # ------------------------------- CONSTRUCTOR :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self, batch_size: int) -> None:

        # ----------------------------------------------------------------------------------
        # ----------------- Gather the Artistic Styles and Images Therefor -----------------
        # ----------------------------------------------------------------------------------
        root = pathlib.Path('/Users/lucasyoung/Desktop/Dataset')

        styles = []
        styles.extend(filter(pathlib.Path.is_dir, (root / 'Paintings').iterdir()))
        styles.extend(filter(pathlib.Path.is_dir, (root / 'Pokemon').iterdir()))
        styles.extend(filter(pathlib.Path.is_dir, (root / 'MtG').iterdir()))

        styles = list(filter(lambda path: len(list(path.iterdir())) >= 16, styles))

        self.styles = {
            style: list(style.rglob('*.png')) for style in styles
        }


        # ----------------------------------------------------------------------------------
        # --------------------------- Initialize Transformations ---------------------------
        # ----------------------------------------------------------------------------------
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(512, interpolation=Image.BILINEAR),
            transforms.RandomCrop(512),
            transforms.ToTensor(),
        ])


        # ----------------------------------------------------------------------------------
        # ----------------------------------- Attributes -----------------------------------
        # ----------------------------------------------------------------------------------
        self.batch_size = batch_size


    # ------------------------------------------------------------------------------------------
    # ---------------------------------- OPERATOR :: Iterator ----------------------------------
    # ------------------------------------------------------------------------------------------
    def __iter__(self) -> typing.Iterator[tuple[pathlib.Path, pathlib.Path]]:
        """Shuffles all styles, selects two distinct pairs, and yields BATCH_SIZE // 2 images at
        random from each."""

        styles = list(enumerate(self.styles.values())) # pair lists of paths with labels
        styles = list(random.sample(styles, len(self.styles))) # shuffle
        zipped = zip(styles, styles[1:] + styles[:1]) # zip with rotation for pairwise iteration

        for (label_a, images_a), (label_b, images_b) in zipped:

            for image_a in random.sample(images_a, self.batch_size // 2):
                yield image_a, label_a

            for image_b in random.sample(images_b, self.batch_size // 2):
                yield image_b, label_b


    # ------------------------------------------------------------------------------------------
    # ----------------------------------- OPERATOR :: Length -----------------------------------
    # ------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.styles)


    # ------------------------------------------------------------------------------------------
    # ---------------------------------- OPERATOR :: Get Item ----------------------------------
    # ------------------------------------------------------------------------------------------
    def __getitem__(self, data: tuple[pathlib.Path, int]) -> T_Tensors:
        return self.transforms(Image.open(data[0])), data[1] # image, label