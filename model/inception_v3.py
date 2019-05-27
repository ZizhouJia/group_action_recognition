import torchvision.models as models
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
import torch

class inception_v3(nn.Module):
    def __init__(self,pca_dir="./pca_matrix/yt8m_pca",device_ind  =0):
        super(inception_v3,self).__init__()
        self.base=models.inception_v3(pretrained=True)
        self.cuda_device = torch.device('cuda:' + str(device_ind))
        self.pca_dir=pca_dir
        if(self.pca_dir is not None):
            self._load_pca()
        # print(self.base)
        print('inception establish ')

    def forward(self,x):
        x = self.base.Conv2d_1a_3x3(x)
        x = self.base.Conv2d_2a_3x3(x)
        x = self.base.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.base.Conv2d_3b_1x1(x)
        x = self.base.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.base.Mixed_5b(x)
        x = self.base.Mixed_5c(x)
        x = self.base.Mixed_5d(x)
        x = self.base.Mixed_6a(x)
        x = self.base.Mixed_6b(x)
        x = self.base.Mixed_6c(x)
        x = self.base.Mixed_6d(x)
        x = self.base.Mixed_6e(x)
        x = self.base.Mixed_7a(x)
        x = self.base.Mixed_7b(x)
        x = self.base.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        if(self.pca_dir is not None):
            x = x.squeeze(-1)
            x = x.squeeze(-1)
            x=x-self.pca_mean
            x=x.mm(self.pca_eigenvecs)
            x=x/torch.sqrt(1e-8+self.pca_eigenvals)
        return x

    def _load_pca(self):
        self.pca_mean = np.load(
            os.path.join(self.pca_dir, 'mean.npy'))[:, 0]
        self.pca_mean=torch.Tensor(self.pca_mean).float().cuda(self.cuda_device).view(1,-1)
        self.pca_eigenvals = np.load(
            os.path.join(self.pca_dir, 'eigenvals.npy'))[:1024, 0]
        self.pca_eigenvals=torch.Tensor(self.pca_eigenvals).float().cuda(self.cuda_device).view(1,-1)
        self.pca_eigenvecs = np.load(
            os.path.join(self.pca_dir, 'eigenvecs.npy')).T[:, :1024]
        self.pca_eigenvecs=torch.Tensor(self.pca_eigenvecs).float().cuda(self.cuda_device).view(2048,1024)

