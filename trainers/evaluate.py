import os 
import wandb

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data.sampler import WeightedRandomSampler

from datasets.VideoDataset import VideoDataset
from datasets.TabularDataset import TabularDataset
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
from models.Evaluator import Evaluator
from models.Evaluator_regression import Evaluator_Regression
from utils.utils import grab_arg_from_checkpoint, grab_hard_eval_image_augmentations, grab_wids, create_logdir

from trainers.utils_as import fix_leakage #TODO - change loc

from typing import Dict, Union, List
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

tufts_label_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'binary': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 1, 'severe_AS': 1},
    'mild_mod': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 2, 'severe_AS': 2},
    'mod_severe': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 1, 'severe_AS': 2},
    'four_class': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 2, 'severe_AS': 3},
    'five_class': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 2, 'moderate_AS': 3, 'severe_AS': 4},
}

view_scheme = {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':3, 'A4CorA2CorOther':4}
view_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'three_class': {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':2, 'A4CorA2CorOther':2},
    'four_class': {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':2, 'A4CorA2CorOther':3},
    'five_class': {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':3, 'A4CorA2CorOther':4},
}

#For human reference
class_labels: Dict[str, List[str]] = {
    'binary': ['No AS', 'AS'],
    'mild_mod': ['No AS', 'Early', 'Significant'],
    'mod_severe': ['No AS', 'Mild-mod', 'Severe'],
    'four_class': ['No AS', 'Mild', 'Moderate', 'Severe'],
    'five_class': ['No AS', 'Mild', 'Mild-mod', 'Moderate', 'Severe']
}

def load_private_dataset(hparams):
  
  img_dataset = pd.read_csv(hparams.data_train_imaging)
  img_dataset['path'] = img_dataset['path'].map(lambda x: join(hparams.private_dataset_root, x))

  # remove unnecessary columns in 'as_label' based on label scheme
  label_scheme_name = hparams.label_scheme_name
  scheme = label_schemes[label_scheme_name]
  img_dataset = img_dataset[img_dataset['as_label'].isin( scheme.keys() )]

  # train/val/test sets
  train_df = img_dataset[img_dataset['split'] == 'train']  
  train_df = fix_leakage(df=img_dataset, df_subset=train_df, split='train')

  # train_df = train_df.sample(frac=1)
  # train_df = train_df.iloc[:500] #TODO - remove later
  # print(train_df['as_label'].to_list())

  val_df = img_dataset[img_dataset['split'] == 'val'] 
  val_df = fix_leakage(df=img_dataset, df_subset=val_df, split='val')

  test_df = img_dataset[img_dataset['split'] == 'test'] 
  test_df = fix_leakage(df=img_dataset, df_subset=test_df, split='test') 

  train_dataset = VideoDataset(
    train_df, hparams.eval_train_augment_rate, 
    grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=True, 
                            task=hparams.task, label_scheme=scheme, image_loader=hparams.private_image_loader, dataset_type='Private')

  val_dataset = VideoDataset( 
    val_df, hparams.eval_train_augment_rate, 
    grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=True, 
                            task=hparams.task, label_scheme=scheme, image_loader=hparams.private_image_loader, dataset_type='Private')

  test_dataset = VideoDataset(test_df, 0, grab_arg_from_checkpoint(hparams, 'img_size'), 
                              target=hparams.target, 
                              train=False, 
                              task=hparams.task,
                              label_scheme=scheme,
                              image_loader=hparams.private_image_loader,
                              dataset_type='Private')
  
  return train_dataset, val_dataset, test_dataset


def evaluate_private_dataset(hparams, wandb_logger):
  """
  Evaluates trained contrastive models. 
  
  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger
  """
  pl.seed_everything(hparams.seed)
  
  train_dataset, val_dataset, test_dataset = load_private_dataset(hparams)
  
  drop = ((len(train_dataset)%hparams.batch_size)==1)

  sampler = None
  if hparams.weights:
    print('Using weighted random sampler(')
    weights_list = [hparams.weights[int(l)] for l in train_dataset.labels]
    sampler = WeightedRandomSampler(weights=weights_list, num_samples=len(weights_list), replacement=True)
  
  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=1, sampler=sampler,
    pin_memory=True, shuffle=True, drop_last=drop, persistent_workers=True) #TODO - change shuffle to true

  val_loader = DataLoader( 
    val_dataset,
    num_workers=hparams.num_workers, batch_size=1,
    pin_memory=True, shuffle=False, persistent_workers=True)

  logdir = create_logdir('eval', hparams.resume_training, hparams.wandb_run_name)

  if hparams.task == 'regression':
    model = Evaluator_Regression(hparams)
  else:
    model = Evaluator(hparams)
  
  mode = 'max'
  
  callbacks = []
  callbacks.append(ModelCheckpoint(monitor=f'eval.val.{hparams.eval_metric}', mode=mode, filename=f'checkpoint_best_{hparams.eval_metric}', dirpath=logdir))
  callbacks.append(EarlyStopping(monitor=f'eval.val.{hparams.eval_metric}', min_delta=0.0002, patience=int(10*(1/hparams.val_check_interval)), verbose=False, mode=mode))
  if hparams.use_wandb:
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer.from_argparse_args(hparams, accelerator="gpu", devices=1, callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch, val_check_interval=hparams.val_check_interval, limit_train_batches=hparams.limit_train_batches, limit_val_batches=hparams.limit_val_batches, limit_test_batches=hparams.limit_test_batches)

  trainer.fit(model, train_loader, val_loader) 
  
  wandb_logger.log_metrics({f'best.val.{hparams.eval_metric}': model.best_val_score})

  if hparams.test_and_eval_private:
    hparams.transform_test = test_dataset.transform_val.__repr__()
    drop = ((len(test_dataset)%hparams.batch_size)==1)

    test_loader = DataLoader(
      test_dataset,
      num_workers=hparams.num_workers, batch_size=1,  
      pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True) 

    model.freeze()

    trainer.test(model, test_loader, ckpt_path=join(logdir,f'checkpoint_best_{hparams.eval_metric}.ckpt')) 














def load_tufts_dataset(hparams):
        
  # read in the data directory CSV as a pandas dataframe
  dataset = pd.read_csv(join(hparams.tufts_dataset_root, hparams.tufts_csv_name))
  # append dataset root to each path in the dataframe
  dataset['path'] = dataset.apply(lambda x: join(hparams.tufts_dataset_root, x['SourceFolder'], x['query_key']), axis=1)
  
  if view in ('PLAX', 'PSAX'):
      dataset = dataset[dataset['view_label'] == view]
  elif view == 'plaxpsax':
      dataset = dataset[dataset['view_label'].isin(['PLAX', 'PSAX'])]
  elif view == 'no_other':
      dataset = dataset[dataset['view_label'] != 'A4CorA2CorOther']
  elif view != 'all':
      raise ValueError(f'View should be PLAX/PSAX/PLAXPSAX/no_other/all, got {view}')
  
  # remove unnecessary columns in 'as_label' based on label scheme
  scheme = tufts_label_schemes[hparams.tufts_label_scheme_name]
  scheme_view = view_schemes[view_scheme_name]
  dataset = dataset[dataset['diagnosis_label'].isin( self.scheme.keys() )]
  
  eval_dataset = VideoDataset( 
    dataset, hparams.eval_train_augment_rate, 
    grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=True, 
    task=hparams.task, label_scheme=scheme, image_loader=hparams.tufts_image_loader, dataset_type='Tufts')

  return eval_dataset

def evaluate_tufts_dataset(hparams, wandb_logger):
  pass