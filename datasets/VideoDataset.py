from typing import Tuple
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.io import read_image
from scipy.io import loadmat
from random import lognormvariate, randint
from skimage.transform import resize

from utils.utils import grab_hard_eval_image_augmentations, grab_soft_eval_image_augmentations, grab_image_augmentations

class VideoDataset(Dataset):
  """
  Dataset for the evaluation of images
  """
  def __init__(self, imaging_dataframe, eval_train_augment_rate: float, img_size: int, 
                target: str, train: bool, task: str, label_scheme: dict, image_loader: str, dataset_type: str, 
                hr_mean: float = 4.237, #delete_segmentation: bool, live_loading: bool, 
      hr_std: float = 0.1885) -> None:
    super(VideoDataset, self).__init__()
    self.train = train
    self.eval_train_augment_rate = eval_train_augment_rate
    #self.live_loading = live_loading
    self.task = task
    self.hr_mean = hr_mean
    self.hr_std = hr_std

    self.dataset_type = dataset_type
    self.data = imaging_dataframe
    #self.labels = torch.load(labels)
    self.scheme = label_scheme
    self.img_size = img_size
    self.image_loader = self.get_loader(image_loader)

    # if delete_segmentation:
    #   for im in self.data:
    #     im[0,:,:] = 0

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target)
    self.transform_val = transforms.Compose([
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])
  
  def get_loader(self, loader_name):
    loader_lookup_table = {'mat_loader': self.mat_loader, 'png_loader': self.png_loader}
    return loader_lookup_table[loader_name]

  def mat_loader(self, path):
      return loadmat(path)['cine']

  def png_loader(self, path):
      return plt.imread(path)
  
  @staticmethod
  def get_random_interval(vid, length):
        length = int(length)
        start = randint(0, max(0, len(vid) - length))
        return vid[start:start + length]
  
  @staticmethod
  def gray_to_gray3(in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(-1, 3, -1, -1)

  def __getitem__(self, indx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns an video for evaluation purposes.
    If training, has {eval_train_augment_rate} chance of being augmented.
    If val, never augmented.
    """
    data_info = self.data.iloc[indx]
    if self.dataset_type=='Private':
      cine_original = self.image_loader(data_info['path'])
      window_length = 60000 / (lognormvariate(self.hr_mean, self.hr_std) * data_info['frame_time'])
      cine = self.get_random_interval(cine_original, window_length)
      cine = resize(cine, (32, self.img_size, self.img_size)) 
      cine = torch.tensor(cine).unsqueeze(1) #[f, c, h, w]
      cine = self.gray_to_gray3(cine) #[f, c, h, w]
      # if self.live_loading:
      #   im = read_image(im)
      #   im = im / 255
      transformed_cine = torch.zeros(cine.shape)
      for i in range(cine.shape[0]):
        im = cine[i, :, :, :]
        if self.train and (random.random() <= self.eval_train_augment_rate):
          im = self.transform_train(im.squeeze(0)) 
          transformed_cine[i, :, :, :] = im.unsqueeze(0)
          # if i == 0:
          #   print(f"transformed_cine has transformations: {torch.all(im.unsqueeze(0)==transformed_cine[i, :, :, :])}")
        else:
          im = cine[i, :, :]
          im = self.transform_val(im.squeeze(0)) 
          transformed_cine[i, :, :, :] = im.unsqueeze(0)
      transformed_cine = transformed_cine.float()
      labels_AS = torch.tensor(self.scheme[data_info['as_label']])
      return (transformed_cine), labels_AS
    elif dataset_type=='Tufts':
      #TODO
      pass
    else:
      print(f"Dataset class does not support this type of dataset. It is currently only compatible with Tufts and Private dataset.")


  def __len__(self) -> int:
    return self.data.shape[0]
