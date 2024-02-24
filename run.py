import os 
import sys
import time
import random
from multiprocessing import Queue

import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from trainers.pretrain import pretrain
from trainers.evaluate import evaluate_private_dataset, evaluate_tufts_dataset
from trainers.test import test
from trainers.generate_embeddings import generate_embeddings
from utils.utils import grab_arg_from_checkpoint, prepend_paths, re_prepend_paths

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

#@hydra.main(config_path='./configs', config_name='config', version_base=None)
def run(args: DictConfig):
  pl.seed_everything(args.seed)
  #args = prepend_paths(args)
  #time.sleep(random.randint(1,5)) # Prevents multiple runs getting the same version when launching many jobs at once
  
  args.wandb_run_name = 'hager-pretrain_as_tom' #TODO 
  #args.wandb_run_name = 'hager-eval_as_tom'
  
  base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
  if args.use_wandb:
    wandb_logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, save_dir=base_dir, offline=args.offline)
  else:
    wandb_logger = WandbLogger(project='multimodal_as', entity='andreafung6', name=args.wandb_run_name, save_dir=base_dir, offline=args.offline)
  args.wandb_id = wandb_logger.version

  # if args.checkpoint and not args.resume_training:
  #   if not args.datatype:
  #     args.datatype = grab_arg_from_checkpoint(args, 'datatype')
      
  if args.pretrain:
    pretrain(args, wandb_logger) 
    args.checkpoint = os.path.join(base_dir, 'runs', args.datatype, args.wandb_run_name, f'checkpoint_last_epoch_{args.max_epochs-1:02}.ckpt')
  
  if args.test_private:
    test_private(args, wandb_logger)
  elif args.evaluate_private:
    evaluate_private_dataset(args, wandb_logger)

  if args.test_tufts:
    test_tufts(args, wandb_logger)
  elif args.evaluate_tufts:
    evaluate_tufts_dataset(args, wandb_logger)

  wandb.finish()
  del wandb_logger

@property
def exception(self):
  if self._pconn.poll():
    self._exception = self._pconn.recv()
  return self._exception

@hydra.main(config_path='./configs', config_name='config', version_base=None)
def control(args: DictConfig):
  run(args)

if __name__ == "__main__":
  control()

