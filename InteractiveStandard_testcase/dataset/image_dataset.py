import logging
logging.getLogger().setLevel(logging.INFO)

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from torch.utils.data import Dataset


def prepare_broken_data(
    csv_layout: pd.DataFrame, 
    data_cleaning: Dict[str, int],
) -> pd.DataFrame:
    '''
    Input:
        data_cleaning: dict with threshold indices for cleaning layout. 
            There's specific csv from test case where data is messy
    '''
    new_clusters = defaultdict(list)

    for cluster, thresh_idx in data_cleaning.items():
        images = list(csv_layout[csv_layout.cluster_id == cluster].file_name)
        new_clusters[cluster] = images[:thresh_idx]
        new_clusters['other'].extend(images[thresh_idx:])
    new_data = []
    for cluster_id, images in new_clusters.items():
        new_data.extend([[cluster_id, image] for image in images])
        
    data = pd.DataFrame(new_data, columns = ['cluster_id', 'file_name'])

    return data


def get_train_test_data(
    csv_layout: pd.DataFrame,
    test_size: float = 0.2,
) -> (pd.DataFrame, pd.DataFrame):

    encoder = LabelEncoder()
    csv_layout['label'] = encoder.fit_transform(csv_layout['cluster_id'])

    X = csv_layout[['cluster_id', 'file_name']]
    y = csv_layout['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=42)
    
    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    
    assert len(set(train_df.label).difference(set(test_df.label))) == 0, "Train or test labels have unique value"

    logging.info("Dataset info")
    logging.info(f"Number of classes: {len(train_df.label.unique())}")
    logging.info(f"Size of train data: {train_df.shape}")
    logging.info(f"Size of test data: {test_df.shape}")

    return train_df, test_df


class ImageDataset(Dataset):
    '''
    Input:
        image_dir: Path object to image directory
        csv_layout: csv with layout (columns filename, label required)
        transforms: torchvision.transforms.Compose object for image
        return_labels: whether to return label in __getitem__
        verify_images: whether to verify images
    '''
    
    def __init__(
        self,
        image_dir: Path,
        csv_layout: pd.DataFrame,
        transforms: transforms.Compose = None,
        return_labels=True,
        verify_images=False
    ):
        self.image_dir = image_dir   
        self.csv_layout = csv_layout
        assert self.csv_layout is not None, "There is no csv file" 

        # auxiliary structures 
        self.image_to_label = None
        self.label_to_images = None
        self.csv_to_labels()
        
        self.images = list(self.csv_layout['file_name'])
        self.images_paths = [Path(self.image_dir, image) for image in self.images]
        self.labels = list(self.csv_layout['label']) if 'label' in self.csv_layout else list(self.csv_layout['cluster_id'])
        
        if verify_images:
            self.verify_images()
        
        self.augmentations = transforms
        self.return_labels = return_labels
        
        assert self.labels, "There is no labels in data" 
        
    def csv_to_labels(self) -> None:
        '''
        Сonvert Pandas.DataFrame with label - image columns
        to dict image->label (i.e cluster of image)
        '''
        assert 'file_name' in self.csv_layout, "file_name not in csv_layout" 
        assert 'label' in self.csv_layout or 'cluster_id' in self.csv_layout, "label not in csv_layout" 
        
        self.image_to_label = defaultdict()
        self.label_to_images = defaultdict(list)
        
        clusters = self.csv_layout['label'] if 'label' in self.csv_layout else self.csv_layout['cluster_id']
        images = self.csv_layout['file_name']
        
        for class_id, filename in zip(clusters, images):
            self.image_to_label[filename] = class_id
            self.label_to_images[class_id].append(filename)
            
            
    def verify_images(self) -> None:
        valid_images = 0
        
        logging.info("Start image validation")
        for image_path in tqdm(self.images_paths):
            try:
                image = Image.open(image_path).convert('RGB')
                valid_images += 1
                
            except Exception as e:
                print(f'corrupted image: {image_path}', e)
                
        logging.info(f"Valid data: {valid_images}, {100 * valid_images/len(self.images_paths)}%")
        logging.info(f"Corrupted data: {len(self.images_paths) - valid_images}, {100 * (1 - valid_images/len(self.images_paths))}%")
                
        
    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, idx) -> Union[torch.tensor, List[torch.tensor]]:
        
        image_path = self.images_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.augmentations:
            image = self.augmentations(image)
        
        if self.return_labels:
            label = self.labels[idx]
            return image, torch.tensor(label)
        else:
            return image