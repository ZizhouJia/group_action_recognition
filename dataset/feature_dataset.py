import torch.utils.data as Data
import torch

class feature_dataset(Data.Dataset):
  def __init__(self,features,label,soft=None):
      self.features=features
      self.label=label
      self.soft_label=soft

  def __getitem__(self,index):
     return "0",torch.Tensor(self.features[index,:,:,:]),self.label[index],self.soft_label[index]

  def __len__(self):
     return self.features.shape[0]

