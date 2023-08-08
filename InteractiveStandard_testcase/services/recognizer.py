import logging
logging.getLogger().setLevel(logging.INFO)

from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import torch
import torchvision

from dataset import ImageDataset, get_valid_transforms, prepare_broken_data
from models import EfficientNet
from params import embedding_size, image_size, data_cleaning

device = torch.device('cpu')

class Recognizer():
    def __init__(
        self,
        image_dir: Union[Path, str], 
        csv_path: Union[Path, str],
        trunk_path: Union[Path, str], 
        embedder_path: Union[Path, str]
    ):
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.trunk_path = trunk_path
        self.embedder_path = embedder_path
        self.model = None
        self.data = None
        self.dataset = None
        self.transforms = None
        self.cluster_id_to_label = {}
        self.label_to_cluster_id = {}

        self._load_data()
        self._load_model()


    def _load_data(self):
        self.data = pd.read_csv(self.csv_path)
        encoder = LabelEncoder()
        self.data['label'] = encoder.fit_transform(self.data['cluster_id'])

        classes = encoder.classes_
        labels = encoder.transform(classes)
        for cl, l in zip(classes, labels):
            self.cluster_id_to_label[cl] = l
            self.label_to_cluster_id[l] = cl
        
        
        num_classes = len(self.data.label.unique())
        logging.info(f"Dataset has {num_classes} classes")
        
        self.transforms = get_valid_transforms(image_size)
        self.dataset = ImageDataset(self.image_dir, self.data, self.transforms)

        
    def _load_model(self):
        '''
            Initialize main model with weight, which located at trunk_path and embedder_path.
        '''
        self.model = EfficientNet(embedding_size=embedding_size, data_parallel=False)
        self.model.load(self.trunk_path, self.embedder_path, device=device)
        self.model.set_inference_instance()

    
    def build_and_save_knn_by_data(self, save_knn_path: Union[Path, str]) -> None:
        '''
            Build knn based on image dataset.
            Please, provide csv_path and image_dir for data.
        ''' 
        logging.info("Start training knn...")
        self.model.inference_model.train_knn(self.dataset)
        
        logging.info(f"Knn successully trained, saved file: {str(save_knn_path)}")
        self.model.inference_model.save_knn_func(save_knn_path)


    def load_knn(self, load_knn_path: Union[Path, str]) -> None:
        self.model.inference_model.load_knn_func(load_knn_path)
        logging.info(f"Knn successully loaded")

    def get_label_for_image(
        self, 
        image: Union[str, Path, torch.Tensor], 
        return_dist: bool = False
    ) -> Union[Tuple[int, str], Tuple[float, int, str]]:
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image).convert('RGB')
            image = self.transforms(image)
        
        if image.shape[0] == 3:
            image = image.unsqueeze(0)
            

        distances, indices = self.model.inference_model.get_nearest_neighbors(image, k=1)
        dist = float(distances[0][0])
        index = int(indices[0][0])
        pred_label = self.dataset.labels[index]
        cluster_id = self.label_to_cluster_id[pred_label]

        if return_dist:
            return (dist, pred_label, cluster_id)
        else:
            return (pred_label, cluster_id)


    def get_labels_for_batch(
        self, 
        batch: Union[List[str], List[Path], torch.Tensor], 
        return_dist: bool = False
    ) -> Union[Tuple[List[int], List[str]], Tuple[List[float], List[int], List[str]]]:
        if isinstance(batch, torch.Tensor):
            assert batch.shape[0] == 4, "Expected 4D input tensor"
        else:
            images = []
            for image_path in batch:
                image = Image.open(image_path).convert('RGB')
                image = self.transforms(image)
                images.append(image)
            batch = torch.stack(images)
            

        distances, indices = self.model.inference_model.get_nearest_neighbors(batch, k=1)

        dists = []
        pred_labels = []
        pred_cluster_ids = []
        for d, idx in zip(distances, indices):
            dists.append(float(d[0]))
            
            label = self.dataset.labels[idx[0]]
            pred_labels.append(label)
            pred_cluster_ids.append(self.label_to_cluster_id[label])


        if return_dist:
            return (dists, pred_labels, pred_cluster_ids)
        else:
            return (pred_labels, pred_cluster_ids)

        