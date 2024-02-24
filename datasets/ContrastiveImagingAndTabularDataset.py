from typing import List, Tuple
import random
import csv
import copy

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from torchvision.io import read_image
from scipy.io import loadmat
from random import lognormvariate, randint
from skimage.transform import resize

class ContrastiveImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.

  The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
  The first tabular view is never augmented. The second view is corrupted by replacing {corruption_rate} features
  with values chosen from the empirical marginal distribution of that feature.
  """
  def __init__(
      self, 
      imaging_dataframe, augmentation: transforms.Compose, augmentation_rate: float, 
      tabular_dataframe, corruption_rate: float, one_hot_tabular: bool, #field_lengths_tabular: str, labels_path: str, live_loading: bool, 
      img_size: int, label_scheme, hr_mean: float = 4.237, 
      hr_std: float = 0.1885) -> None:
            
    # Imaging
    self.data_imaging = imaging_dataframe
    self.transform = augmentation
    self.augmentation_rate = augmentation_rate
    #self.live_loading = live_loading
    self.hr_mean = hr_mean
    self.hr_srd = hr_std
    self.img_size = img_size

    # if self.delete_segmentation:
    #   for im in self.data_imaging:
    #     im[0,:,:] = 0

    self.default_transform = transforms.Compose([
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])

    # Tabular
    self.data_tabular = tabular_dataframe
    self.generate_marginal_distributions(tabular_dataframe)
    self.c = corruption_rate
    #self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.one_hot_tabular = one_hot_tabular
    
    # Classifier
    #self.labels = torch.load(labels_path)
    self.scheme = label_scheme

  def generate_marginal_distributions(self, data) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    self.marginal_distributions = data.transpose().values.tolist()

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    # if self.one_hot_tabular:
    #   return int(sum(self.field_lengths_tabular))
    # else:
    #   return len(self.data[0])
    return len(self.data_tabular.iloc[0])

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
    return subject

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def mat_loader(self, path):
    mat = loadmat(path)
    if 'cine' in mat.keys():    
        return loadmat(path)['cine']
    if 'cropped' in mat.keys():    
        return loadmat(path)['cropped']

  @staticmethod
  def get_random_interval(vid, length):
        length = int(length)
        start = randint(0, max(0, len(vid) - length))
        return vid[start:start + length]
  
  @staticmethod
  def gray_to_gray3(in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(-1, 3, -1, -1)

  def generate_imaging_views(self, im) -> List[torch.Tensor]:
    """
    Generates two views of a subjects image. Also returns original image resized to required dimensions.
    The first is always augmented. The second has {augmentation_rate} chance to be augmented.
    """
    
    # if self.live_loading:
    #   im = read_image(im)
    #   im = im / 255
    ims = [self.transform(im)]
    if random.random() < self.augmentation_rate:
      ims.append(self.transform(im))
    else:
      ims.append(self.default_transform(im))

    orig_im = self.default_transform(im)
    return ims, orig_im

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    data_info = self.data_imaging.iloc[index]
    study_num = data_info['Echo ID#']
    labels_AS = torch.tensor(self.scheme[data_info['as_label']])

    cine_original = self.mat_loader(data_info['path'])
    window_length = 60000 / (lognormvariate(self.hr_mean, self.hr_srd) * data_info['frame_time'])
    cine = self.get_random_interval(cine_original, window_length)
    frame_choice = np.random.randint(0, cine.shape[0], 1)
    cine = cine[frame_choice, :, :]
    cine = resize(cine, (1, self.img_size, self.img_size)) 

    cine = torch.tensor(cine).unsqueeze(1) #[f, c, h, w]
    cine = self.gray_to_gray3(cine)
    cine = cine.squeeze(0) #[c, h, w]
    cine = cine.float()

    imaging_views, unaugmented_image = self.generate_imaging_views(im=cine)
    tabular_views = [torch.tensor(self.data_tabular.loc[int(study_num)], dtype=torch.float), torch.tensor(self.corrupt(self.data_tabular.loc[int(study_num)]), dtype=torch.float)]
    if self.one_hot_tabular:
      tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]
    
    #return imaging_views, tabular_views, label, unaugmented_image
    return imaging_views, tabular_views, labels_AS, unaugmented_image

  def __len__(self) -> int:
    return len(self.data_imaging)