import logging
logging.getLogger().setLevel(logging.INFO)

from pathlib import Path
from typing import Tuple

import ipywidgets as widgets
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from torchvision import transforms

from .image_dataset import ImageDataset


def get_images_from_class(dataset: ImageDataset, class_id: int) -> None:
    '''
    Draw all images from class in Jupyter Notebook
    '''
    images = dataset.label_to_images[class_id]
    image_path = dataset.image_dir
    pairs = np.array_split(images, int(len(images) / 2))

    for pair in pairs:
        path1 = Path(image_path, pair[0])
        path2 = Path(image_path, pair[1])
        
        image1 = open(path1, 'rb').read()
        image2 = open(path2, 'rb').read()

        wi1 = widgets.Image(value=image1, format='jpg', width=300, height=400)
        wi2 = widgets.Image(value=image2, format='jpg', width=300, height=400)
        a = [wi1, wi2]
        wid = widgets.HBox(a)
        logging.info(f"Image path 1: {path1}")
        logging.info(f"Image path 2: {path2}")
        display.display(wid)


def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, epoch, *args):
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    plt.figure(figsize=(10, 7))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
        
    plt.savefig(f"{split_name}_{keyname}_{epoch}.png")
    plt.show()



def get_train_transforms(image_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
            [
                transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                
            ]
    )

def get_valid_transforms(image_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
            [
                transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),                
            ]
    )
