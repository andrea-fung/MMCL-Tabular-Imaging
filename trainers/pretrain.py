import os 
import sys

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.utils import grab_image_augmentations, grab_wids, create_logdir
from utils.ssl_online_custom import SSLOnlineEvaluator

from datasets.ContrastiveImagingAndTabularDataset import ContrastiveImagingAndTabularDataset
from datasets.ContrastiveImageDataset import ContrastiveImageDataset
from datasets.ContrastiveTabularDataset import ContrastiveTabularDataset

from models.MultimodalSimCLR import MultimodalSimCLR
from models.SimCLR import SimCLR
from models.SwAV_Bolt import SwAV
from models.BYOL_Bolt import BYOL
from models.SimSiam_Bolt import SimSiam
from models.BarlowTwins import BarlowTwins
from models.SCARF import SCARF

from trainers.utils_as import load_as_data, preprocess_as_data, fix_leakage #TODO - change loc

from typing import Dict, Union
from random import lognormvariate
from os.path import join

import numpy as np
import pandas as pd

label_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'binary': {'normal': 0.0, 'mild': 1.0, 'moderate': 1.0, 'severe': 1.0},
    'all': {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3},
    'not_severe': {'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 1},
    'as_only': {'mild': 0, 'moderate': 1, 'severe': 2},
    'mild_moderate': {'mild': 0, 'moderate': 1},
    'moderate_severe': {'moderate': 0, 'severe': 1}
}

def load_datasets(hparams):
  transform = grab_image_augmentations(hparams.img_size, hparams.target) 
  hparams.transform = transform.__repr__()

  img_dataset = pd.read_csv(hparams.data_train_imaging)
  img_dataset['path'] = img_dataset['path'].map(lambda x: join(hparams.private_dataset_root, x))

  # remove unnecessary columns in 'as_label' based on label scheme
  label_scheme_name = hparams.label_scheme_name
  scheme = label_schemes[label_scheme_name]
  img_dataset = img_dataset[img_dataset['as_label'].isin( scheme.keys() )]

  # train/val/test sets
  train_df = img_dataset[img_dataset['split'] == 'train']  
  train_df = fix_leakage(df=img_dataset, df_subset=train_df, split='train')

  val_df = img_dataset[img_dataset['split'] == 'val'] 
  val_df = fix_leakage(df=img_dataset, df_subset=val_df, split='val')

  test_df = img_dataset[img_dataset['split'] == 'test'] 
  test_df = fix_leakage(df=img_dataset, df_subset=test_df, split='test') 

  # get tabular data
  tab_train, tab_val, tab_test = load_as_data(csv_path = hparams.data_train_tabular,
                                              drop_cols = [],
                                              num_ex = None,
                                              scale_feats = True)

  #perform imputation 
  tab_train, tab_val, tab_test, _ = preprocess_as_data(tab_train, tab_val, tab_test, cat_cols=[])

  train_dataset = ContrastiveImagingAndTabularDataset(
    train_df, transform, hparams.augmentation_rate, 
    tab_train, hparams.corruption_rate, hparams.one_hot, #hparams.field_lengths_tabular, hparams.labels_train, hparams.live_loading
    hparams.img_size, scheme)

  val_dataset = ContrastiveImagingAndTabularDataset(
    val_df, transform, hparams.augmentation_rate, 
    tab_val, hparams.corruption_rate, hparams.one_hot, #hparams.field_lengths_tabular, hparams.labels_train, hparams.live_loading
    hparams.img_size, scheme)

  test_dataset = ContrastiveImagingAndTabularDataset(
    test_df, transform, hparams.augmentation_rate, 
    tab_test, hparams.corruption_rate, hparams.one_hot, #hparams.field_lengths_tabular, hparams.labels_train, hparams.live_loading
    hparams.img_size, scheme)

  hparams.input_size = train_dataset.get_input_size()
  
  return train_dataset, val_dataset, test_dataset

def pretrain(hparams, wandb_logger):
  """
  Train code for pretraining or supervised models. 
  
  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger
  """
  pl.seed_everything(hparams.seed)

  # Load appropriate dataset
  train_dataset, val_dataset, test_dataset = load_datasets(hparams)
  
  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=True, drop_last=True, persistent_workers=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, drop_last=True, persistent_workers=True)

  test_loader = DataLoader(
    test_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, drop_last=True, persistent_workers=True)

  # Create logdir based on WandB run name
  logdir = create_logdir(hparams.datatype, hparams.resume_training, hparams.wandb_run_name) 
  
  model = MultimodalSimCLR(hparams)
  
  callbacks = []

  if hparams.online_mlp: 
    model.hparams.classifier_freq = float('Inf')
    callbacks.append(SSLOnlineEvaluator(z_dim = model.pooled_dim, hidden_dim = hparams.embedding_dim, num_classes = hparams.num_classes, swav = False, multimodal = (hparams.datatype=='multimodal')))
  callbacks.append(ModelCheckpoint(filename='checkpoint_{epoch:02d}-{classifier.val.acc:.2f}', 
                                    dirpath=logdir, 
                                    monitor='classifier.val.acc',
                                    mode='max',
                                    save_on_train_epoch_end=True, 
                                    auto_insert_metric_name=False))
  callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer.from_argparse_args(hparams,
                                        gpus=hparams.num_gpus, 
                                        callbacks=callbacks, 
                                        logger=wandb_logger, 
                                        max_epochs=hparams.max_epochs, 
                                        check_val_every_n_epoch=hparams.check_val_every_n_epoch, 
                                        limit_train_batches=hparams.limit_train_batches, 
                                        limit_val_batches=hparams.limit_val_batches, 
                                        enable_progress_bar=hparams.enable_progress_bar)

  trainer.fit(model, train_loader, val_loader)