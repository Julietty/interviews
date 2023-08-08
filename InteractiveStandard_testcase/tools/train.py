import umap.umap_ as umap

import pandas as pd

import torch

from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils import logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from dataset import (
    ImageDataset, 
    prepare_broken_data, 
    get_train_test_data, 
    get_train_transforms, get_valid_transforms, 
    visualizer_hook
)
from models import EfficientNet
from params import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # Data prepare
    data = pd.read_csv(csv_dir, index_col='Unnamed: 0')
    # Delete when data is good
    data = prepare_broken_data(data, data_cleaning)
    train_df, test_df = get_train_test_data(data, test_size=0.3)
    test_df, val_df = get_train_test_data(data, test_size=0.4)
    train_df = train_df.to_csv('train_df.csv')
    test_df = test_df.to_csv('test_df.csv')
    val_df = val_df.to_csv('val_df.csv')
    num_classes = len(train_df.label.unique())

    train_transforms = get_train_transforms(image_size)
    test_transforms = get_valid_transforms(image_size)

    train_dataset = ImageDataset(image_dir, train_df, train_transforms)
    test_dataset = ImageDataset(image_dir, test_df, test_transforms)
    
    # Set model
    model = EfficientNet(device, embedding_size)
    
    # Set optimizers
    trunk_optimizer = torch.optim.Adam(model.trunk.parameters(), lr=lr, weight_decay=weight_decay)
    embedder_optimizer = torch.optim.Adam(model.embedder.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Set loss:
    # For person recognition arcface loss is nice
    loss = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size, 
                              margin=arcface_margin, scale=arcface_scale)
    
    # Pairs miner
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    
    # Sampler
    sampler = samplers.MPerClassSampler(
        train_dataset.labels, m=samples_per_class, length_before_new_iter=len(train_dataset)
    )
    
    # Structures for Trainer
    models = {
        "trunk": model.trunk, 
        "embedder": model.embedder
    }
    optimizers = {
        "trunk_optimizer": trunk_optimizer,
        "embedder_optimizer": embedder_optimizer,
    }
    
    loss_funcs = {"metric_loss": loss}
    
    # Set logger
    record_keeper, _, _ = logging_presets.get_record_keeper("logs", "tensorboard")
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"val": test_dataset}
    
    # Hooks for on-training validation and visualization 
    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        visualizer=umap.UMAP(),
        visualizer_hook=visualizer_hook,
        dataloader_num_workers=dataloader_num_workers,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )
    
    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester, dataset_dict, model_folder, test_interval=1, patience=1
    )
    
    trainer = trainers.MetricLossOnly(
        models=models,
        optimizers=optimizers,
        batch_size=batch_size,
        loss_funcs=loss_funcs,
        dataset=train_dataset,
        mining_funcs={},
        sampler=sampler,
        dataloader_num_workers=dataloader_num_workers,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )
    
    # Start training
    trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    train()